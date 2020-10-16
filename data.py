import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class TrainDataset(object):
    def __init__(self, image_dir, csv_path, train_type, batch_size, random_seed, shuffle):
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.train_type = train_type
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.shuffle= shuffle
        self.num_classes = 0
        self.dataframe = pd.read_csv(self.csv_path, sep='\t', index_col=None)
        self.datagen = ImageDataGenerator(
            rotation_range=30.,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            rescale=1./255,
            validation_split=0.2)

        self.train_generator, self.validation_generator = self.__create_train_valid_gen()

    def __create_train_valid_gen(self):
        __y_cols = [col for col in self.dataframe.columns if self.train_type in col]
        self.num_classes = len(__y_cols)

        train_generator = self.datagen.flow_from_dataframe(
            dataframe=self.dataframe,
            directory='data/',
            x_col='image_name',
            y_col = __y_cols,
            subset='training',
            batch_size=self.batch_size,
            seed=self.random_seed,
            shuffle=self.shuffle,
            class_mode="raw",
            target_size=(192, 192))

        valid_generator = self.datagen.flow_from_dataframe(
            dataframe=self.dataframe,
            directory='data/',
            x_col='image_name',
            y_col = __y_cols,
            subset='validation',
            batch_size=self.batch_size,
            seed=self.random_seed,
            shuffle=self.shuffle,
            class_mode="raw",
            target_size=(192, 192))

        return train_generator, valid_generator


if __name__ == '__main__':
    train_dataset = TrainDataset(
        image_dir='data/',
        csv_path='data/dataset_csv/list_combined_attr1_small.tsv',
        train_type='attribute1',
        batch_size=32,
        shuffle=True,
        random_seed=10
    )
