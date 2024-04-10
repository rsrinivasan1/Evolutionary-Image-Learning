from .shape import Shape
import cv2
import numpy as np

class Circle(Shape):
    def __init__(self, original, canvas, coords, size, color=None):
        super().__init__(original, canvas, coords, size, color)
        self.alpha = 0.8
        self.color = None
        self.alpha_color = None
        self.set_color(color)
        self.pixels_under = self.get_pixels_under()

    def set_color(self, color):
        self.color = color
        if color is not None:
            self.alpha_color = (color[0] * self.alpha,
                            color[1] * self.alpha,
                            color[2] * self.alpha)

    def update(self):
        overlay = self.canvas.copy()
        cv2.circle(overlay, (self.x, self.y), self.size, self.color, -1)
        cv2.addWeighted(overlay, self.alpha, self.canvas, 1 - self.alpha, 0, self.canvas)

    def get_pixels_under(self):
        """
        :return: Return a list of pixels and their colors in the form [(x, y, (R, G, B)), ...]
        that this circle will write over
        """
        circle_pixels = []
        for x in range(self.x - self.size, self.x + self.size + 1):
            for y in range(self.y - self.size, self.y + self.size + 1):
                # if coords within circle radius and in image bounds
                if (x - self.x) ** 2 + (y - self.y) ** 2 <= self.size ** 2 and \
                        0 <= x < len(self.canvas) and 0 <= y < len(self.canvas[0]):
                    circle_pixels.append((x, y, self.canvas[y][x]))
        return circle_pixels

    def get_average_color_under(self):
        all_colors = np.array([0, 0, 0])
        total = 0
        for x in range(self.x - self.size, self.x + self.size + 1):
            for y in range(self.y - self.size, self.y + self.size + 1):
                # if coords within circle radius and in image bounds
                if (x - self.x) ** 2 + (y - self.y) ** 2 <= self.size ** 2 and \
                        0 <= x < len(self.canvas) and 0 <= y < len(self.canvas[0]):
                    all_colors += self.canvas[y][x]
                    total += 1
        return all_colors // total

    def loss_before(self):
        before = 0
        for x, y, color in self.pixels_under:
            before += np.sum((color - self.original[y][x]) ** 2)
        return before

    def loss_after(self):
        after = 0
        for x, y, color in self.pixels_under:
            new_color = self.alpha_color + color * (1 - self.alpha)
            after += np.sum((new_color - self.original[y][x]) ** 2)
        return after

    def __str__(self):
        return f"Circle at {self.x}, {self.y} - radius {self.size}, color: {self.color}"
