import os
from glob import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.transform import resize


class Inference(object):
    def __init__(self, model_path: str, sample_dir: str, inference_csv: str, inference_type: str):
        self.model_path = model_path
        self.inference_csv = inference_csv
        self.inference_type = inference_type
        self.sample_dir = sample_dir

        self.model = None
        self.__get_model()

        self.images = self.__load_samples()
        self.labels = self.__get_labels()

    def __get_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def __get_labels(self):
        total_labels = np.squeeze(pd.read_csv(self.inference_csv).values)
        # labels = [l for l in total_labels if self.inference_type in l]
        return np.array(total_labels)

    def __load_samples(self):
        files = glob(os.path.join(self.sample_dir, '**.jpg'))
        images = []
        for f in files:
            im = imread(f)
            im = resize(im, output_shape=(224, 224))
            images.append(im)

        images = np.array(images)
        if len(images.shape) == 3:
            images = images[np.newaxis, : ,: ,:]

        return np.squeeze(images)

    def predict(self, save_result):
        total_predictions = []
        for i, im in enumerate(self.images):
            im = im[np.newaxis, :, :, :]
            prediction = self.model.predict(im).reshape(-1)
            total_predictions.append(prediction)
            len_classes = 46
            indexes = np.array([i for i in range(len_classes)])
            sorted_to_best = [indexes for prediction, indexes in sorted(zip(prediction, indexes), key=lambda pair: pair[0])]

            if save_result:
                fig, axs = plt.subplots(nrows=2, ncols=1)
                axs[0].imshow(im[0, :, :, :])
                axs[1].barh(self.labels[sorted_to_best[-5:]], prediction[sorted_to_best[-5::]], align='center')
                fig.savefig(f'samples/pred/sample_pred{i}.png', dpi=400)
        return total_predictions


if __name__ == '__main__':
    inf = Inference(model_path=f'models/category.h5',
                    sample_dir='samples',
                    inference_type='category',
                    inference_csv=f'data/category.csv')

    inf.predict(save_result=True)