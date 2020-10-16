import tensorflow as tf


class Trainer(object):
    def __init__(self, model: tf.keras.models.Model, train_gen: tf.keras.preprocessing.image.ImageDataGenerator,
                 val_gen: tf.keras.preprocessing.image.ImageDataGenerator, epoch: int):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.epoch = epoch

        self.step_size_train = self.train_gen.n // self.train_gen.batch_size
        self.step_size_val = self.val_gen.n // self.val_gen.batch_size

    def train(self):
        self.model.fit(
            x=self.train_gen,
            steps_per_epoch=self.step_size_train,
            validation_data=self.val_gen,
            validation_steps=self.step_size_val,
            epochs=self.epoch,
        )
