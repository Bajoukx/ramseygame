"""Learns the environment"""

from absl import app
from absl import logging
from absl import flags

from ramsey_env import RamseyGame
from stable_baselines3 import PPO

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_nodes', 6, 'Number of Nodes', lower_bound=0)

flags.DEFINE_integer('k_clique_number',
                     3,
                     'Size of clique to find in graph.',
                     lower_bound=3)

flags.DEFINE_integer('n_timesteps',
                     10000,
                     'Number of timesteps to run the nevironment for',
                     lower_bound=1)


def main(_):
    """Learns the environment."""

    environment = RamseyGame(n_nodes=FLAGS.n_nodes,
                             k_clique=FLAGS.k_clique_number)
    environment.reset()

    model = PPO('MlpPolicy', environment, verbose=1, learning_rate=0.001)

    while True:
        model.learn(total_timesteps=FLAGS.n_timesteps)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
