import gym
import gym_battleship

env = gym.make('battleship-v0')
env.reset()

for i in range(10):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(f'Reward for step {i}: {reward}')
