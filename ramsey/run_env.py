import time
import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ramsey_env import RamseyGame
current_time = time.time()
models_dir = f'models/{int(current_time)}/'
logdir = f'logs/{int(current_time)}/'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(models_dir):
    os.makedirs(logdir)

environment = RamseyGame(n_nodes = 6, k_clique = 3)
environment.reset()

model = PPO('MlpPolicy', environment, verbose=1, tensorboard_log=logdir)

# TIMESTEPS = 10000
# iters = 0

# while True:
#     iters += 1
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f'PPO')
#     model.save(f'{models_dir}/{TIMESTEPS*iters}')
episodes = 3
for episode in range(episodes):

    done = False
    obs = environment.reset()
    while not done:
        random_action = environment.action_space.sample()
        print('action', random_action)
        obs, reward, done, info = environment.step(random_action)
        print('reward', reward)
        environment.render()

        if done:
            environment.render()
        