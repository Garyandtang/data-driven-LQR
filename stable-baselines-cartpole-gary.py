import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


from envs.cartpole.cartpole import CartPole

env = gym.make("Cartpole1-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

vec_env = CartPole(gui=True)
obs, _ = vec_env.reset()
while 1:
    action, _states = model.predict(obs, deterministic=True)
    print("action: ", action)
    obs, reward, done, _, _ = vec_env.step(action)
    print("state: ", obs)
env.close()