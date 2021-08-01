# CartPole Challenge

This is an OpenAI challenge. View more [here](https://gym.openai.com/envs/CartPole-v1/).

Description of the problem accordingly to OpenAI itself:

> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.


Here is an example of an execution

![](cartpole.gif?raw=true)

## Evaluation and Results

We generate 1000 games where the agent make aleatory decisions using a threshold T [10, 20, 30, 40, 50], whenever we hit a score above T, we save that game to train the agent afterwards. This process repeats 100 times for each approach we will be testing. The score for an approach is the average of the scores an agent managed to reach after being trained using the data from the aleatory games saved previously. 

Here are the results:

![](results.png?raw=true)