# Evolutionary Image Learning

<img width="1000" alt="tiger" src="https://github.com/rsrinivasan1/Evolutionary-Image-Learning/assets/52140136/b5565293-1a72-4a89-af5e-4e55625c949d">

## Methods

In this project, I sought to use evolutionary methods to estimate an RGB image, inspired by the [Circle Evolution](https://github.com/ahmedkhalf/Circle-Evolution/) project by Ahmed Khalf. To do this, we start by randomly throwing colored circles onto a blank canvas. Those that improve the image loss by a certain threshold are retained, so that new circles are placed on top of old ones.

In future iterations, when we pick an (x, y) coordinate to place a colored circle, the color distribution is centered about the average color of placed pixels (not of the original image) surrounding that (x, y) coordinate. In this way, this evolutionary algorithm is able to tend toward the correct RGB values at each (x, y) coordinate. Each randomly generated circle that reduces the loss swings the color distribution closer to the true color, and eventually we end up with an image resembling the original.

--------
## Reinforcement Learning Trials

When starting this project, I originally wanted to learn how to use Proximal Policy Optimization (PPO), and aimed to use RL techniques to produce a policy network that can replicate the correct RGB values at each coordinate based on getting rewards for correct values. This method eventually ended up not working, and I have a few hypotheses why:

1. Predicting the correct colors at every coordinate in an image involves a near _continuous action space_. Since there are 256^3 possibilities for an RGB color, and a reinforcement learning algorithm gets rewarded based on guessing the correct color for a specific (x, y) coordinate, getting a reward for a specific action is hard to interpret. Did the algorithm get a reward out of pure luck, or did it guess the right color for that pixel? It's hard to know, and might not be enough for the algorithm to converge.

2. Even when the algorithm gets a reward, it's hard to know if that is the _best_ reward possible. For instance, let's say I place circle R:200, G:100, B:50 at position (75, 25), with radius 5. In return, I get reward 10. Is this the best reward I can get at this pixel location? What if another color gets an even better reward? Since the algorithm doesn't know, I hypothesize that it is settling for suboptimal rewards just because they are better than nothing.

3. The policy network may not have enough complexity to learn the intricate details of every pixel on an image. Since I was running this project on my local machine, I may not have had enough computing power to properly have the algorithm learn the best pixel colors at every coordinate.

These are just my best guesses, and I am still learning. I recognize that this is not a great problem to try to solve using RL, but I would like to use this as a learning opportunity. Please email me at rsrinivasan2021@gmail.com if you have any ideas on how to make this work!

--------
## Other sample images from evolution

![mona_lisa](https://github.com/rsrinivasan1/Evolutionary-Image-Learning/assets/52140136/c48b664c-ff75-4724-ae49-7306d903a705)
![raghav](https://github.com/rsrinivasan1/Evolutionary-Image-Learning/assets/52140136/e5a95a3c-dd86-4d5a-beb7-6678cd1bcda7)
