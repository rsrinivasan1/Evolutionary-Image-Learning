from runner import Runner
from shapes.shape_types import ShapeTypes

if __name__ == "__main__":
    # runner = Runner("white.png", ShapeTypes.CIRCLE, "evo")
    runner = Runner("white.png", ShapeTypes.CIRCLE, "rl")
    runner.train()
    runner.render()
