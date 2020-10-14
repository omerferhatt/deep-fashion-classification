import tensorflow as tf


class Inference(object):
    def __init__(self, model_path: str, sample_dir: str, sample_csv:str):
        self.model_path = model_path
        self.sample_dir = sample_dir
        self.sample_csv = sample_csv

        self.model = self.__get_model()

    def __get_model(self):
        return tf.keras.models.load_model(self.model_path)

    def __load_samples(self):
        pass
