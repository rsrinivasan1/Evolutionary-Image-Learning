import numpy as np
from shapes.shape_types import ShapeTypes
from shapes.circle import Circle

class Agent:
    def __init__(self, original, width, height, shape_type, runner):
        self.original = original.astype(np.float32)
        self.width = width
        self.height = height
        self.canvas = 255 * np.ones((self.height, self.width, 3), dtype=np.uint8)
        self.shape_type = shape_type
        self.runner = runner

    def compute_loss(self):
        return np.sum((self.original - self.canvas) ** 2)

    def create_shape(self, coords, size, canvas=None, color=None):
        if canvas is None:
            canvas = self.canvas
        if self.shape_type == ShapeTypes.CIRCLE:
            shape = Circle(self.original, canvas, coords, size, color=color)
            return shape
