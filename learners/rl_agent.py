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
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

        # self.policy_network = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # Added max pooling with kernel size 2 and stride 2
        #     nn.ReLU(),
        #     nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear((width // 2) * (height // 2), 4),  # output size is 4 for the specified means and variances
        #     nn.LayerNorm(4),
        #     nn.Sigmoid()
        # )

    def activation_to_range(self, x, min_val, max_val):
        return x * (max_val - min_val) + min_val

    def get_denormalized_action(self, action):
        _, r, g, b = action
        return (self.width // 10,
                self.activation_to_range(r, 0, 255),
                self.activation_to_range(g, 0, 255),
                self.activation_to_range(b, 0, 255))

    def get_reward_from_action(self, y, x, action, canvas):
        radius, r, g, b = self.get_denormalized_action(action)
        shape = self.create_shape((x, y), radius, canvas, torch.stack([b, g, r]))
        # loss_before = shape.loss_before()
        # loss_after = shape.loss_after_torch()

        average_color = shape.get_average_color_under_original()

        # perform action
        shape.update(pytorch=True)
        # Return high reward for low loss
        return 0, (r - average_color[2]) ** 2 + (g - average_color[1]) ** 2 + (b - average_color[0]) ** 2

    def get_action_distribution(self, y, x):
        y = y / self.height
        x = x / self.width
        action_means_variances = self.policy_network(torch.tensor([y, y, y, y, y, y, y, y, y, y,
                                                                   x, x, x, x, x, x, x, x, x, x]).float())
        # action_means = torch.stack([action_means_variances[0],
        #                             action_means_variances[1],
        #                             action_means_variances[2],
        #                             action_means_variances[3]])
        # action_variances = nn.ReLU()(torch.stack([action_means_variances[4] * 0.2 + 0.001,
        #                                           action_means_variances[5] * 0.2 + 0.001,
        #                                           action_means_variances[6] * 0.2 + 0.001,
        #                                           action_means_variances[7] * 0.2 + 0.001]))

        # canvas = torch.zeros(1, self.height, self.width)
        # canvas[0, y, x] = 1
        # radius = self.width // 10
        # grid_y, grid_x = torch.meshgrid(torch.arange(canvas.size(1)), torch.arange(canvas.size(2)))
        # distances = torch.sqrt((grid_y - y) ** 2 + (grid_x - x) ** 2)

        # Create a mask for pixels within the radius
        # mask = distances <= radius
        # canvas[0, mask] = 1
        #
        # action_means_variances = self.policy_network(canvas)
        # action_means = torch.stack([action_means_variances[0][0],
        #                             action_means_variances[0][1],
        #                             action_means_variances[0][2],
        #                             action_means_variances[0][3]])
        # action_variances = nn.ReLU()(torch.stack([action_means_variances[0][4] * 0.1,
        #                                           action_means_variances[0][5] * 0.1,
        #                                           action_means_variances[0][6] * 0.1,
        #                                           action_means_variances[0][7] * 0.1]))

        # action_distribution = torch.distributions.Normal(action_means, action_variances)
        # return action_distribution
        return action_means_variances

    def get_action_and_log_prob(self, y, x):
        action_distribution = self.get_action_distribution(y, x)
        # orig_action = action_distribution.rsample()
        # log_prob = action_distribution.log_prob(orig_action)

        # return orig_action, log_prob
        return action_distribution, 0

    def train(self):
        """
        NOTES:
        Update policy network after T timesteps, for each actor
        """
        optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=0.01)

        start = time()
        initial = self.compute_loss()
        loss = initial

        for i in range(100000):

            # Get log probability of previous action using the current policy network
            # y_coord, x_coord = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            j = i // 500
            num = self.width // 10
            y_coord, x_coord = (j % (self.height * self.width // (num ** 2))) // (self.width // num), (j % (self.height * self.width // (num ** 2))) % (self.width // num)
            y_coord, x_coord = y_coord * num, x_coord * num
            if i % 2 == 0:
                y_coord, x_coord = 10, 160
            else:
                y_coord, x_coord = 10, 120

            action_t, log_prob = self.get_action_and_log_prob(y_coord, x_coord)

            before, after = self.get_reward_from_action(y_coord, x_coord, action_t, self.canvas)
            loss += after - before

            # 3. Compute advantage estimate — here a state has high advantage if the loss is low
            if i % 400 == 0:
                print(f"Loss: ")
                print(after)
                print(f"Log prob: ")
                print(log_prob)
                # policy_loss = p1 * torch.exp(log_prob[1]) + p2 * torch.exp(log_prob[2]) + p3 * torch.exp(log_prob[3])
                policy_loss = after
                print(f"Policy loss:")
                print(policy_loss)

                # initial_state_dict = copy.deepcopy(self.policy_network.state_dict())

                optimizer_policy.zero_grad()
                policy_loss.backward()
                optimizer_policy.step()

                # distribution = self.get_action_distribution(y_coord, x_coord)
                # print(f"Action distribution: {distribution.loc, distribution.scale}")

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
