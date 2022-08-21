"""Learns the environment"""

from absl import app
from absl import logging
from absl import flags

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

import ramsey  # pylint: disable=unused-import
from ramsey.configs.graph import GraphConfig
from ramsey.envs import utils

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

flags.DEFINE_boolean(
    'save_counterexample', False,
    'Whether to save the counterexample. A counterexample is a graph (and \
    it\'s dual) that does not have a clique of size k_clique_number.')


def main(_):
    """Learns the environment."""

    graph_config = GraphConfig(n_nodes=FLAGS.n_nodes,
                               k_clique=FLAGS.k_clique_number,
                               save_counterexample=FLAGS.save_counterexample)
    print('graph_config:', graph_config)

    env = utils.make_environment('RamseyGame-v1', seed=42, graph_config=graph_config)()
    #env = gym.make('RamseyGame-v1', config=graph_config)
    #env = Monitor(env)
    #env.seed(42)
    #env.reset()
    #print(check_env(env))
    check_env(env)

    #model = A2C('MlpPolicy', env, verbose=1)

    #model.learn(total_timesteps=FLAGS.n_timesteps)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
