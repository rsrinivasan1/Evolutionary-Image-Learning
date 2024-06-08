from .agent import Agent
from random import randint
from time import time
import numpy as np

class EvoAgent(Agent):
    """Try throwing paint at the wall and see if it sticks"""

    def __init__(self, original, width, height, shape_type, runner):
        super().__init__(original, width, height, shape_type, runner)
        self.initial_sector_loss = {}

    def get_sector(self, num_divisions, i):
        """
            Get boundaries of a specific region of an image
        """
        i = i % (num_divisions ** 2)
        x_start = (i % num_divisions) * (self.width // num_divisions)
        x_end = ((i % num_divisions) + 1) * (self.width // num_divisions)
        y_start = max(0, int((i // num_divisions) * (self.height / num_divisions)))
        y_end = min(self.height, int((i // num_divisions + 1) * (self.height / num_divisions)))
        return x_start, x_end, y_start, y_end

    def continue_sector(self, i, x_start, x_end, y_start, y_end):
        """
            Determine whether this region of the image is close enough to the original
            so that we can stop placing pixels here
        """
        sector = self.original[y_start:y_end, x_start:x_end]

        if i in self.initial_sector_loss:
            initial_loss = self.initial_sector_loss[i]
        else:
            white_canvas = np.full_like(sector, 255)
            initial_loss = np.sum((sector - white_canvas) ** 2)
            self.initial_sector_loss[i] = initial_loss

        curr_sector = self.canvas[y_start:y_end, x_start:x_end]
        sector_loss = np.sum((sector - curr_sector) ** 2)
        if initial_loss == 0:
            return False
        accuracy = (initial_loss - sector_loss) / initial_loss
        return accuracy < 0.996

    def train(self):
        start = time()
        initial = self.compute_loss()
        loss = initial
        num_sectors = 20
        for i in range(100000):
            # try different mutations
            x_start, x_end, y_start, y_end = self.get_sector(num_sectors, i)
            before = 0
            after = 1
            found = True
            j = 0
            best_loss = 0
            best_shape = None
            while after >= before * 0.8:
                if j == 100 and not self.continue_sector(i % (num_sectors ** 2), x_start, x_end, y_start, y_end):
                    found = False
                    break
                if j == 200:
                    if best_shape is None:
                        found = False
                    break
                x = randint(x_start, x_end)
                y = randint(y_start, y_end)
                radius = randint(3, self.width // 32)
                shape = self.create_shape((x, y), radius)
                color = shape.get_average_color_under()
                shape.set_color((min(255, max(0, randint(color[0] - 60, color[0] + 60))),
                                min(255, max(0, randint(color[1] - 60, color[1] + 60))),
                                min(255, max(0, randint(color[2] - 60, color[2] + 60)))))
                before = shape.loss_before()
                after = shape.loss_after()
                if after - before < best_loss:
                    best_loss = after - before
                    best_shape = shape
                j += 1

            # we found a good mutation, update state
            if found:
                best_shape.update()
                loss += best_loss

            if i % 100 == 0:
                end = time()
                print(f"Iteration {i} - {(end - start) / 60} min; loss - {loss}")
                print(f"Completion: {(initial - loss) / initial * 100}")
                self.runner.render()
