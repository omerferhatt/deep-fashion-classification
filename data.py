import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


class TrainDataset(object):
    def __init__(self, image_dir, csv_path_train, csv_path_val, train_type, batch_size, random_seed, shuffle, image_shape):
        self.image_dir = image_dir
        self.csv_path_train = csv_path_train
        self.csv_path_val = csv_path_val
        self.train_type = train_type
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.image_shape = image_shape

        self.num_classes = 0
        self.dataframe_train = pd.read_csv(self.csv_path_train, sep='\t', index_col=None)
        self.dataframe_val = pd.read_csv(self.csv_path_val, sep='\t', index_col=None)

        self.datagen_train = ImageDataGenerator(
            rotation_range=25.,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            preprocessing_function=preprocess_input,
            rescale=1./255)

        self.datagen_val = ImageDataGenerator(rescale=1./255)

        self.train_generator, self.validation_generator = self.__create_train_valid_gen()

    def __create_train_valid_gen(self):
        __y_cols = [col for col in self.dataframe_train.columns if self.train_type in col]
        self.num_classes = len(__y_cols)

        train_generator = self.datagen_train.flow_from_dataframe(
            dataframe=self.dataframe_train,
            directory='data/',
            x_col='image_name',
            y_col = __y_cols,
            batch_size=self.batch_size,
            seed=self.random_seed,
            shuffle=self.shuffle,
            class_mode="raw",
            target_size=self.image_shape)

        valid_generator = self.datagen_val.flow_from_dataframe(
            dataframe=self.dataframe_val,
            directory='data/',
            x_col='image_name',
            y_col = __y_cols,
            batch_size=self.batch_size,
            seed=self.random_seed,
            shuffle=self.shuffle,
            class_mode="raw",
            target_size=self.image_shape)

        return train_generator, valid_generator


if __name__ == '__main__':
    train_dataset = TrainDataset(
        image_dir='data/',
        csv_path_train='data/dataset_csv/list_combined_category_small_train.tsv',
        csv_path_val='data/dataset_csv/list_combined_category_small_val.tsv',
        train_type='category',
        batch_size=32,
        shuffle=True,
        random_seed=10,
        image_shape=(224, 224)
    )
