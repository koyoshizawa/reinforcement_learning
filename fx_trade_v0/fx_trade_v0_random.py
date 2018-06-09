import myenv
import gym
import random

env = gym.make('myenv-v0')

Episodes = 1

obs = []

for _ in range(Episodes):
    observation = env.reset()
    done = False
    count = 0

    while not done:
        action = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3])
        observation, reward, done, info = env.step(action)
        obs = obs + [observation]
        count += 1


        if done:
            print('reward: ', reward)
            print('steps: ', count)