from absl import app
from absl import logging
import numpy as np

from ramsey.dataset import generator


def main(_):
    dataset = generator.CustomGraphDataset(43, 0.5, n_samples=2000)
    logging.info(dataset[100]['clique_number'])
    file_path = 'test_'
    batch = []
    for count in range(len(dataset)):
        batch.append(dataset[count])
        if not (count + 1) % 1000:
            logging.info(f'Saved samples: {count}/{len(dataset)}.')
            with open((file_path + str(count + 1) + '.npy'), 'wb') as file:
                np.save(file, batch)
            batch = []

if __name__ == '__main__':
    app.run(main)
