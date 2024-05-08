from .agent import Agent
import torch
from torch import nn, optim

class RLAgent(Agent):
    """Wander in some direction and learn from successes & failures

    NOTE: this is a non-sparse reward setting, since our actions of placing pixels
    have immediate rewards/penalties
    - near-continuous action space

    State: self.canvas
    Output: action, described as (x, y, size, color)

    Notes on policy gradient methods:
    - Value function: estimate of discounted rewards in future
        - allows agent to learn what states are advantageous
        - might learn to "setup" for next placement
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
    """
    def __init__(self, original, width, height, shape_type, runner):
        super().__init__(original, width, height, shape_type, runner)
        # Policy network:
        #   - input: self.canvas
        #   - output: mean x, mean y, mean radius, mean r, mean g, mean b
        #             variance x, variance y, variance radius, variance r, variance g, variance b

        # Policy network using CNN:
        self.policy_network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width * height, 512),  # Adjust input size based on width and height
            nn.ReLU(),
            nn.Linear(512, 12),  # output size is 12 for the specified means and variances
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
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width * height, 128),  # Adjust input size based on width and height
            nn.ReLU(),
            nn.Linear(128, 1),  # output size is 12 for the specified means and variances
        )

    def sigmoid_to_range(self, x, min_val, max_val):
        return int(x * (max_val - min_val) + min_val)

    def get_denormalized_action(self, action):
        x, y, radius, r, g, b = action
        return (self.sigmoid_to_range(x, 0, self.width),
                self.sigmoid_to_range(y, 0, self.height),
                self.sigmoid_to_range(radius, 3, self.width // 32),
                self.sigmoid_to_range(r, 0, 255),
                self.sigmoid_to_range(g, 0, 255),
                self.sigmoid_to_range(b, 0, 255))

    def get_reward_from_action(self, action, canvas):
        x, y, radius, r, g, b = self.get_denormalized_action(action)
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

    def get_r_theta(self):
        return 0

    def train(self):
        num_actors = 1
        T = 10
        optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=0.001)
        optimizer_value = optim.Adam(self.value_network.parameters(), lr=0.001)
        epsilon = 0.2  # PPO clip ratio
        discount_factor = 0.95
        for i in range(10000):
            for j in range(num_actors):
                # 1. For each actor, run the current policy for T timesteps
                # actual_rewards contains the undiscounted rewards from each timestep
                actual_rewards = [0]
                # estimated_rewards contains the DISCOUNTED total reward estimate — this
                # is what we try to learn
                estimated_rewards = [0]
                # copy canvas and move RGB channel dimension to front
                actor_canvas = self.canvas.copy()
                torch_canvas = torch.moveaxis(torch.from_numpy(actor_canvas).float(), 2, 0)

                for k in range(T):
                    action_means_variances = self.policy_network(torch_canvas)
                    action_means, action_variances = torch.split(action_means_variances, 6, dim=1)  # Split means and variances

                    action_distribution = torch.distributions.Normal(action_means[0], action_variances[0])
                    action = torch.clamp(action_distribution.sample(), min=0)  # Sample action from the distribution

                    actual_rewards.append(actual_rewards[-1] + (discount_factor ** k) * self.get_reward_from_action(action, actor_canvas))

                    # Get expected rewards from value network
                    estimated_rewards.append(self.value_network(torch_canvas))

                # 2. Compute advantage estimates (ignore first dummy element)
                advantages = [actual_rewards[i] - estimated_rewards[i] for i in range(1, T + 1)]
                A = sum(ad ** 2 for ad in advantages)
                r_theta = self.get_r_theta()
