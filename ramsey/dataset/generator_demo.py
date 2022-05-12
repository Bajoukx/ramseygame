"""A demo of the ramsey dataset."""

from absl import app
from absl import flags
from absl import logging
import torch

from ramsey.dataset.generator import RamseyGraphDataset

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_nodes', 6, 'Number of Nodes', lower_bound=0)


def log_one_sample(dataset):
    """Logs the data of one element of the dataset."""
    return logging.info(next(iter(dataset)))


def main(_):
    """Logs the data of one dataset element."""
    dataset = RamseyGraphDataset(FLAGS.n_nodes, 0.5, n_samples=1)

    log_one_sample(dataset)

if __name__ == '__main__':
    app.run(main)
