from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self, original, canvas, coords, size, color=None):
        self.original = original
        self.canvas = canvas
        self.x, self.y = coords
        self.size = size
        self.color = color

    @abstractmethod
    def update(self):
        pass
