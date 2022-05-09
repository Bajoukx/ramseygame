import statistics

from absl import app
from absl import logging
import numpy as np

from ramsey.dataset import generator


def main(_):
    file_path = 'test_1000.npy'
    dataset = np.load(file_path, allow_pickle=True)
    logging.info(dataset)
    clique_sizes = []
    dual_clique_sizes = []
    size_one_cliques = []
    size_two_cliques = []
    size_three_cliques = []
    size_four_cliques = []
    for i in range(len(dataset)):
        clique_sizes.append(dataset[i].get('clique_number'))
        dual_clique_sizes.append(dataset[i].get('dual_clique_number'))
    logging.info(statistics.mean(clique_sizes))
    logging.info(statistics.mean(dual_clique_sizes))


if __name__ == '__main__':
    app.run(main)