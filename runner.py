import cv2
from matplotlib import pyplot as plt
import numpy as np

from shapes.shape_types import ShapeTypes
from shapes.circle import Circle
# from learners.rl_agent import RLAgent
from learners.evo_agent import EvoAgent

class Runner:
    def __init__(self, image_name, shape_type, agent):
        self.original = cv2.imread(f'images/{image_name}')
        self.image_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.height, self.width, _ = self.original.shape
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 5))
        self.shape_type = shape_type

        if agent == "evo":
            self.agent = EvoAgent(self.original, self.width, self.height, self.shape_type, self)
        # else:
        #     self.agent = RLAgent(self.original, self.width, self.height, self.shape_type)

        plt.tight_layout()
        self.render_original()

    def render_original(self):
        self.axes[0].imshow(self.image_rgb)

    def render_learned(self):
        image_rgb = cv2.cvtColor(self.agent.canvas, cv2.COLOR_BGR2RGB)
        self.axes[1].imshow(image_rgb)

    def render(self):
        self.render_learned()
        self.fig.show()

    def train(self):
        print("Training...")
        self.agent.train()
