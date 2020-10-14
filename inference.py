import os
from glob import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.transform import resize


class Inference(object):
    def __init__(self, model_path: str, sample_dir: str, inference_csv: str, sample_csv: str, inference_type: str):
        self.model_path = model_path
        self.inference_csv = inference_csv
        self.inference_type = inference_type
        self.sample_dir = sample_dir
        self.sample_csv = sample_csv

        self.model = tf.keras.models.Model()
        self.__get_model()

        self.images = self.__load_samples()
        self.labels = self.__get_labels()

    def __get_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def __get_labels(self):
        total_labels = pd.read_csv(self.inference_csv, sep='\t', header=None, nrows=1)
        labels = [l for l in total_labels if self.inference_type in l]
        return np.array(labels)

    def __load_samples(self):
        files = glob(os.path.join(self.sample_dir, '**.png'))
        images = []
        for f in files:
            im = imread(f)
            im = resize(im, output_shape=(128, 128))
            images.append(im)

        images = np.array(images)
        if len(images.shape) == 4:
            images = images[np.newaxis, : ,: ,:]

        return images

    def predict(self, save_result):
        for im in self.images:
            prediction = self.model.predict(im)
            len_classes = 46
            indexes = [i for i in range(len_classes)]
            sorted_to_best = [indexes for _, indexes in sorted(zip(prediction, indexes))]

            if save_result:
                fig, axs = plt.subplots(nrows=2, ncols=1, sharex='True')
                axs[0].imshow(im)
                axs[1].barh(self.labels[indexes[:-5]], prediction[indexes[:-5]], align='center')


        return prediction

