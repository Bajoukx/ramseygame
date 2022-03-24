import statistics

from absl import app
from absl import logging
import numpy as np

from ramsey.dataset import generator


def main(_):
    dataset = generator.CustomGraphDataset(43, 0.5, n_samples=2000)
    # logging.info(dataset[100]['clique_number'])
    file_path = 'test_1000.npy'
    dataset = np.load(file_path, allow_pickle=True)
    logging.info(dataset)
    clique_sizes = []
    dual_clique_sizes = []
    for i in range(len(dataset)):
        clique_sizes.append(dataset[i].get('clique_number'))
        dual_clique_sizes.append(dataset[i].get('dual_clique_number'))
    logging.info(statistics.mean(clique_sizes))
    logging.info(statistics.mean(dual_clique_sizes))


if __name__ == '__main__':
    app.run(main)
