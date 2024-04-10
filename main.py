from runner import Runner
from shapes.shape_types import ShapeTypes

if __name__ == "__main__":
    runner = Runner("tiger.png", ShapeTypes.CIRCLE, "evo")
    runner.train()
    runner.render()
