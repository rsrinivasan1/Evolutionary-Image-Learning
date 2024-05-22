from .agent import Agent
import torch
from torch import nn, optim
from time import time
import copy
import random

torch.autograd.set_detect_anomaly(True)


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
        #   - input: x, y (normalized to [0, 1])
        #   - output: mean radius, mean r, mean g, mean b
        #             variance radius, variance r, variance g, variance b
        self.policy_network = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width * height, 8),  # output size is 8 for the specified means and variances
            nn.LayerNorm(8),
            nn.Sigmoid()  # activation for bounding output between 0 and 1
        )

    def activation_to_range(self, x, min_val, max_val):
        return int(x * (max_val - min_val) + min_val)

    def get_denormalized_action(self, action):
        radius, r, g, b = action
        # return (self.relu_to_range(radius, 3, max(5, self.width // 32)),
        #         self.relu_to_range(r, 0, 255),
        #         self.relu_to_range(g, 0, 255),
        #         self.relu_to_range(b, 0, 255))
        return (self.width // 10,
                self.activation_to_range(r, 0, 255),
                self.activation_to_range(g, 0, 255),
                self.activation_to_range(b, 0, 255))

    def get_reward_from_action(self, y, x, action, canvas):
        radius, r, g, b = self.get_denormalized_action(action)
        shape = self.create_shape((x, y), radius, canvas, (b, g, r))
        loss_after = shape.loss_after()
        # perform action
        shape.update()
        # Return high reward for low loss
        return -loss_after

    def get_action_distribution(self, y, x):
        canvas = torch.zeros(1, self.height, self.width)
        radius = self.width // 10
        grid_y, grid_x = torch.meshgrid(torch.arange(canvas.size(1)), torch.arange(canvas.size(2)))
        distances = torch.sqrt((grid_y - y) ** 2 + (grid_x - x) ** 2)

        # Create a mask for pixels within the radius
        mask = distances <= radius
        canvas[0, mask] = 1

        action_means_variances = self.policy_network(canvas)
        action_means = torch.stack([action_means_variances[0][0],
                                    action_means_variances[0][1],
                                    action_means_variances[0][2],
                                    action_means_variances[0][3]])
        action_variances = nn.ReLU()(torch.stack([action_means_variances[0][4] * 0.2 + 0.1,
                                                  action_means_variances[0][5] * 0.2 + 0.1,
                                                  action_means_variances[0][6] * 0.2 + 0.1,
                                                  action_means_variances[0][7] * 0.2 + 0.1]))

        action_distribution = torch.distributions.Normal(action_means, action_variances)
        return action_distribution

    def get_action_and_log_prob(self, y, x):
        action_distribution = self.get_action_distribution(y, x)
        orig_action = action_distribution.sample()
        log_prob = action_distribution.log_prob(orig_action).sum()

        return orig_action.detach(), log_prob.detach()

    def train(self):
        """
        NOTES:
        Update policy network after T timesteps, for each actor
        """
        num_actors = 1
        T = 1
        optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=0.1)
        epsilon = 0.2  # PPO clip ratio
        # Save the previous policy's log probabilities for the next update
        next_coords = []
        prev_actions = []
        prev_log_probs = []

        start = time()
        initial = self.compute_loss()

        for i in range(1000000):
            loss = None
            for j in range(num_actors):
                loss = initial

                advantages = []
                curr_log_probs = []
                # 1. Run the current policy for T timesteps
                for t in range(T):
                    # If on first iteration, we don't have an old policy, we move to the next step
                    if next_coords == []:
                        break

                    # Get log probability of previous action using the current policy network
                    y_coord, x_coord = next_coords[t]

                    action_t = prev_actions[t]
                    action_distribution = self.get_action_distribution(y_coord, x_coord)
                    log_prob = action_distribution.log_prob(action_t).sum()
                    curr_log_probs.append(log_prob)

                    if i % 100 == 0 and j == 0:
                        print(f"Action distribution: {action_distribution.loc, action_distribution.scale}")

                    action = torch.clamp(action_t, min=0, max=1)  # Bound action because of variance
                    reward = self.get_reward_from_action(y_coord, x_coord, action, self.canvas)
                    loss -= reward
                    advantages.append(0.9 ** max(t - 1, 0) * reward)

                # 2. Prepare random x, y coords for after we update the policy, and get probability
                # values to calculate r_theta for each of the T actions
                old_log_probs = prev_log_probs
                next_coords = []
                prev_actions = []
                prev_log_probs = []
                for _ in range(T):
                    # y, x = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
                    y, x = 20, 20
                    next_coords.append((y, x))
                    action, log_prob = self.get_action_and_log_prob(y, x)
                    prev_actions.append(action)
                    prev_log_probs.append(log_prob)

                # 3. Compute advantage estimate — here a state has high advantage if the loss is low
                if len(curr_log_probs) != 0 and i % 1000 == 0 and j == 0:
                    # list of prob differences for each sampled action through T timesteps
                    differences = torch.stack(curr_log_probs) - torch.stack(old_log_probs)
                    r_theta = torch.exp(differences)
                    # print(f"R_theta: {r_theta}")
                    # print(f"Curr log probs: {curr_log_probs}")
                    # print(f"Old log probs: {old_log_probs}")
                    advantages = torch.tensor(advantages)
                    # print(f"Advantages: {advantages}")

                    term1 = advantages * r_theta
                    term2 = torch.clamp(r_theta, 1 - epsilon, 1 + epsilon) * advantages

                    policy_loss = -torch.min(term1, term2).mean()
                    print(f"Loss: {policy_loss}")

                    # initial_state_dict = copy.deepcopy(self.policy_network.state_dict())

                    optimizer_policy.zero_grad()
                    policy_loss.backward()
                    optimizer_policy.step()

                    if i % 1000 == 0 and j == 0:
                        end = time()
                        print(f"Iteration {i} - {(end - start) / 60} min; loss - {loss}")
                        print(f"Completion: {(initial - loss) / initial * 100}")
                        self.runner.render()

                    # current_state_dict = self.policy_network.state_dict()
                    # parameters_changed = False
                    # for name, param in initial_state_dict.items():
                    #     if not torch.equal(param, current_state_dict[name]):
                    #         parameters_changed = True
                    #         break
                    #
                    # # Print a message if parameters have changed
                    # if parameters_changed:
                    #     print("Policy network parameters have changed.")
                    # else:
                    #     print("Policy network parameters have not changed.")

            # only thing we can learn in our network is the probability distribution of actions for a given input
