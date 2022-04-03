"""A demo of the ramsey dataset."""

from absl import app
from absl import flags
from absl import logging

from ramsey.dataset.generator import CustomGraphDataset

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_nodes', 6, 'Number of Nodes', lower_bound=0)


def main(_):
    """Logs the data of one dataset element."""
    dataset = CustomGraphDataset(FLAGS.n_nodes, 0.5, n_samples=1)
    
    logging.info('Graph: %s', dataset[0]['graph'])
    logging.info('Clique number: %s', dataset[0]['clique_number'])
    logging.info('Dual: %s', dataset[0]['dual'])
    logging.info('Dual clique number: %s', dataset[0]['dual_clique_number'])
    logging.info('Graph cliques: %s', dataset[0]['cliques'])
    logging.info('Dual cliques: %s', dataset[0]['dual_cliques'])


if __name__ == '__main__':
    app.run(main)
