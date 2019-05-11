import time
import numpy as np
import gym

env = gym.make("CartPole-v0")

max_steps = 200
num_actions = 2

def model_decision(model):
    def decision(prev_obs):
        prediction = model.predict([prev_obs])[0]
        return prediction
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
        observation, reward, done, _ = env.step(int(round(action)))
        # The previous observation led to this action
        if len(prev_obs) > 0:
            memory.append([prev_obs, action])
        prev_obs = observation
        score += reward

        if done:
            break

    return memory, score

def populate(decide=random_decision(env), requirement=50, max_results=-1, remove=0):
    global_score = 0
    x = []
    y = []
    games = 0
    scores = []
    done = False
    while not done:
        memory, score = game(decide)
        scores.append(score)
        global_score += score
        games += 1
        if games % 1000 == 0:
            print("Num of games: " + str(games))
            print("Results size: " + str(len(y)))
        if score >= requirement:
            iterations = max(len(memory) - remove, 0)
            if iterations != 0:
                for episode in range(iterations):
                    x.append(memory[episode][0])
                    y.append(memory[episode][1])
                    if len(y) == max_results:
                        done = True
                        break

    average_score = global_score / games

    return x, y, scores

def test(decide=random_decision(env), visualize=False, games=1000):
    global_score = 0
    for i in range(games):
        if i % 250 == 0:
            print("----Game " + str(i))
        _, score = game(decide)
        global_score += score
        if visualize:
            time.sleep(1)

    average_score = global_score / games
    return average_score