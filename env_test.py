import gym
import gym_battleship

env = gym.make('battleship-v0')
observation = env.reset()

observation, reward, done, info = env.step(env.action_space.sample())
