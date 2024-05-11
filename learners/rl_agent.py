from .agent import Agent
import torch
from torch import nn, optim
import random

class RLAgent(Agent):
    """Wander in some direction and learn from successes & failures

    NOTE: this is a non-sparse reward setting, since our actions of placing pixels
    have immediate rewards/penalties
    - near-continuous action space

    State: self.canvas, (x, y) to place circle
    Output: action, described as (size, color)

    Notes on policy gradient methods:
    - Value function: estimate of discounted rewards in future
        - allows agent to learn what states are advantageous
        - might learn to "setup" for next placement --> this is why each actor has T timesteps to work
        with the current policy
        - input: state, output: estimated discounted reward
    - Advantage estimate (A): discounted reward - value function
        - measure of whether action was better than expected
    - Objective: E[π(a | s) * A]

    - TRPO takes vanilla policy gradient and makes sure that policy doesn't stray too far away
      from previous value -- removes the chance of overly aggressive updates

    Notes on PPO:
    - Main objective: E[min(r(theta) * A, clip(r(theta), 1 - epsilon, 1 + epsilon) * A]
        - clipping makes sure we don't change prob(action) too much:
            - if it's already good, and action is more likely now, don't increase
              prob(action) too much on a single estimate
            - if it's already good, and action is less likely now, make it more likely
            - if it's bad, and action is less likely now, don't decrease prob(action)
              too much because it could just be noise
            - if it's bad, and action is more likely now, undo this by making
              action less likely -- reflected in objective

    - Q. Is T = 1 in this case since we don't need to run our policy for multiple timesteps
         in order to get a reward?
    - A. I think not necessarily, since running the same policy for multiple timesteps will
         always give us a more comprehensive overview of what the policy is doing

    - Actors - entities executing the same policy in the environment for T timesteps
    - Discounted rewards = weighted sum of rewards accumulated after trying the current policy
      for each of t timesteps, for t < T
    - Value function = neural network
        - input: current state of canvas
        - output: estimate of discounted reward (compare with true value to train)
        - objective: sum over all actors over all timesteps of (actual reward - value function output)^2

    IDEA: choose x, y coordinates and make the network learn colors and sizes
    """
    def __init__(self, original, width, height, shape_type, runner):
        super().__init__(original, width, height, shape_type, runner)
        # Policy network:
        #   - input: self.canvas, x, y (as 4th channel)
        #   - output: mean radius, mean r, mean g, mean b
        #             variance radius, variance r, variance g, variance b

        # Policy network using CNN:
        # 4th channel is 1 if pixel is the center for the circle to be placed
        # 0 otherwise
        self.policy_network = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width * height, 512),  # Adjust input size based on width and height
            nn.ReLU(),
            nn.Linear(512, 8),  # output size is 8 for the specified means and variances
            nn.Sigmoid()  # activation for bounding output between 0 and 1
        )
        # Value function network:
        #   - input: self.canvas
        #   - output: estimate of discounted reward
        self.value_network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width * height, 128),  # Adjust input size based on width and height
            nn.ReLU(),
            nn.Linear(128, 1),  # output size is 12 for the specified means and variances
        )

    def sigmoid_to_range(self, x, min_val, max_val):
        return int(x * (max_val - min_val) + min_val)

    def get_denormalized_action(self, action):
        radius, r, g, b = action
        return (self.sigmoid_to_range(radius, 3, self.width // 32),
                self.sigmoid_to_range(r, 0, 255),
                self.sigmoid_to_range(g, 0, 255),
                self.sigmoid_to_range(b, 0, 255))

    def get_reward_from_action(self, y, x, action, canvas):
        radius, r, g, b = self.get_denormalized_action(action)
        print(x, y, radius, r, g, b)
        shape = self.create_shape((x, y), radius, canvas, (r, g, b))
        before = shape.loss_before()
        after = shape.loss_after()
        # perform action
        shape.update()
        # Return positive reward for reduced loss
        # High loss before - low loss after = big reward
        reward = before - after
        return reward

    def get_action_distribution(self, y_x_coords, policy_canvas):
        y_coord, x_coord = y_x_coords

        policy_canvas[0][y_coord][x_coord] = 1

        action_means_variances = self.policy_network(policy_canvas)
        action_means, action_variances = torch.split(action_means_variances, 4, dim=1)  # Split means and variances

        # reset
        policy_canvas[0][y_coord][x_coord] = 0

        # Take element 0 because of how pytorch works
        action_distribution = torch.distributions.Normal(action_means[0], action_variances[0])

        return action_distribution

    def get_action_and_log_prob(self, y_x_coords, policy_canvas):
        action_distribution = self.get_action_distribution(y_x_coords, policy_canvas)
        orig_action = action_distribution.sample()
        log_prob = action_distribution.log_prob(orig_action).sum()

        return orig_action, log_prob

    def train(self):
        """
        NOTES:
        Update policy and value networks after T timesteps, for each actor
        :return:
        """
        num_actors = 1
        T = 10
        optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=0.001, eps=10 ** -5)
        optimizer_value = optim.Adam(self.value_network.parameters(), lr=0.001, eps=10 ** -5)
        epsilon = 0.2  # PPO clip ratio
        discount_factor = 0.9
        # Save the previous policy's log probabilities for the next update
        next_coords = []
        prev_actions = []
        for i in range(1000):
            for j in range(num_actors):
                # actual_rewards contains the discounted rewards from each timestep
                actual_rewards = [0]
                # estimated_rewards contains the discounted total reward estimate — this
                # is what we try to learn
                estimated_rewards = [0]
                # copy canvas and move RGB channel dimension to front
                # torch_canvas now has dimensions 3 x height x width
                actor_canvas = self.canvas.copy()
                value_canvas = torch.moveaxis(torch.from_numpy(actor_canvas).float(), 2, 0)
                # add new dimension to tell network where we plan to center the circle
                policy_canvas = torch.cat((torch.zeros(1, self.height, self.width), value_canvas), dim=0)

                curr_log_probs = []
                # 1. Run the current policy for T timesteps
                for t in range(T):
                    # If first iteration, we don't have an old policy, so we move straight to getting
                    # log probabilities for the next actions
                    if next_coords == []:
                        break

                    # Get log probability of previous action using the current policy network
                    y_coord, x_coord = next_coords[t]
                    action_t = prev_actions[t]
                    action_distribution = self.get_action_distribution((y_coord, x_coord), policy_canvas)
                    log_prob = action_distribution.log_prob(action_t).sum()
                    curr_log_probs.append(log_prob)

                    action = torch.clamp(action_t, min=0)  # Make action non-negative

                    actual_rewards.append(actual_rewards[-1] + (discount_factor ** (min(t-1, 0))) * self.get_reward_from_action(y_coord, x_coord, action, actor_canvas))

                    # Get expected rewards from value network
                    estimated_rewards.append(self.value_network(value_canvas))

                # 2. Prepare random x, y coords for after we update the policy, and get probability
                # values to calculate r_theta for each of the T actions
                next_coords = []
                prev_actions = []
                prev_log_probs = []
                for _ in range(T):
                    y_x_coords = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
                    next_coords.append(y_x_coords)
                    action, log_prob = self.get_action_and_log_prob(y_x_coords, policy_canvas)
                    prev_actions.append(action)
                    prev_log_probs.append(log_prob)

                # 3. Compute advantage estimates (ignore first dummy element)
                if curr_log_probs != []:
                    advantages = torch.tensor([actual_rewards[i] - estimated_rewards[i] for i in range(1, T + 1)])
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-8)

                    r_theta = torch.exp(torch.tensor(curr_log_probs) - torch.tensor(prev_log_probs))
