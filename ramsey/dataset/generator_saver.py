"""Premiliminary template of dataset generator saver."""

from absl import app
from absl import flags
from absl import logging
import numpy as np

from ramsey.dataset import generator

FLAGS = flags.FLAGS

flags.DEFINE_string('file_path', 'data', 'File path to save the dataset.')


def main(_):
    dataset = generator.RamseyGraphDataset(6, 0.5, n_samples=2000)
    logging.info(dataset[100]['clique_number'])
    batch = []
    for count in range(len(dataset)):
        batch.append(dataset[count])
        if not (count + 1) % 1000:
            logging.info(f'Saved samples: {count}/{len(dataset)}.')
            with open((FLAGS.file_path + str(count + 1) + '.npy'),
                      'wb') as file:
                np.save(file, batch)
            batch = []


if __name__ == '__main__':
    app.run(main)
