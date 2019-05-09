import time
import numpy as np
import gym

env = gym.make("CartPole-v0")

max_steps = 200
num_actions = 2

def model_decision(model):
    def decision(prev_obs):
        return np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
    return decision

def random_decision(environment):
    def decision(_):
        return environment.action_space.sample()
    return decision

def game(decide, visualize=False):
    score = 0
    memory = []
    prev_obs = []
    env.reset()
    for _ in range(max_steps):
        if visualize:
            time.sleep(0.02)
            env.render()

        if len(prev_obs) == 0:
            action = env.action_space.sample()
        else:
            action = decide(prev_obs)
        observation, reward, done, _ = env.step(action)
        # The previous observation led to this action
        if len(prev_obs) > 0:
            memory.append([prev_obs, action])
        prev_obs = observation
        score += reward

        if done:
            break

    return memory, score

def populate(decide=random_decision(env), requirement=50, games=1000):
    global_score = 0
    training = []
    for _ in range(games):
        memory, score = game(decide)
        global_score += score
        if score >= requirement:
            for episode in memory:
                output = np.zeros(num_actions)
                output[episode[1]] = 1
                training.append([episode[0], output])

    average_score = global_score / games
    print('Average Score:', average_score)
    return training, average_score

def test(decide=random_decision(env), visualize=False, games=1000):
    global_score = 0
    for _ in range(games):
        _, score = game(decide)
        global_score += score
        if visualize:
            time.sleep(1)

    average_score = global_score / games
    print('Average Score:', average_score)
    return average_score