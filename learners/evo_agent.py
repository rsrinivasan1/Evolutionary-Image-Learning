import numpy as np
from .agent import Agent
from random import randint, random
from time import time


class EvoAgent(Agent):
    """Try throwing paint at the wall and see if it sticks"""

    def __init__(self, original, width, height, shape_type, runner):
        super().__init__(original, width, height, shape_type)
        self.last = []
        self.runner = runner

    def get_sector(self, num_divisions, i):
        i = i % (num_divisions ** 2)
        x_start = (i % num_divisions) * (self.width // num_divisions)
        x_end = ((i % num_divisions) + 1) * (self.width // num_divisions)
        y_start = (i // num_divisions) * (self.height // num_divisions)
        y_end = (i // num_divisions + 1) * (self.height // num_divisions)
        return x_start, x_end, y_start, y_end

    def train(self):
        start = time()
        initial = self.compute_loss()
        loss = initial
        for i in range(500000):
            # try different mutations
            x_start, x_end, y_start, y_end = self.get_sector(10, i)
            before = 0
            after = 1
            shape = None
            while after >= before * 0.8:
                x = randint(x_start, x_end)
                y = randint(y_start, y_end)
                radius = randint(3, self.width // 32)
                shape = self.create_shape((x, y), radius)
                color = shape.get_average_color_under()
                shape.set_color((min(255, max(0, randint(color[0] - 50, color[0] + 50))),
                                min(255, max(0, randint(color[1] - 50, color[1] + 50))),
                                min(255, max(0, randint(color[2] - 50, color[2] + 50)))))
                before = shape.loss_before()
                after = shape.loss_after()

            # we found a good mutation, update state
            shape.update()
            loss += after - before

            if i % 100 == 0:
                end = time()
                print(f"Iteration {i} - {(end - start) / 60} min; loss - {loss}")
                print(f"Completion: {(initial - loss) / initial * 100}")
                self.runner.render()
