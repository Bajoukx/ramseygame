"""Learns the environment"""

import multiprocessing

from absl import app
from absl import logging
from absl import flags

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import ramsey

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_nodes', 6, 'Number of Nodes', lower_bound=0)

flags.DEFINE_integer('k_clique_number',
                     3,
                     'Size of clique to find in graph.',
                     lower_bound=2)

flags.DEFINE_integer('n_timesteps',
                     10000000,
                     'Number of timesteps to run the nevironment for',
                     lower_bound=1)


def make_environment(environment_id, seed, n_nodes, k_clique):
    """Returns a function that creates the environment."""
    def get_env():
        """Returns a environment."""
        env = gym.make(environment_id, n_nodes=n_nodes, k_clique=k_clique)
        env = Monitor(env)
        env.seed(seed)
        env.reset()
        return env

    return get_env


def main(_):
    """Learns the environment."""

    cpu_count = multiprocessing.cpu_count()
    env_list = [
        make_environment('RamseyGame-v0', seed, FLAGS.n_nodes,
                         FLAGS.k_clique_number) for seed in range(cpu_count)
    ]
    environment = SubprocVecEnv(env_list)

    model = A2C('MlpPolicy', environment, verbose=1)

    model.learn(total_timesteps=FLAGS.n_timesteps)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
