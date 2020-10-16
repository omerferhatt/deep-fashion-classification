import tensorflow as tf
import os
import datetime


class Trainer(object):
    def __init__(self, model: tf.keras.models.Model, train_gen: tf.keras.preprocessing.image.ImageDataGenerator,
                 val_gen: tf.keras.preprocessing.image.ImageDataGenerator, epoch: int):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.epoch = epoch

        self.step_size_train = self.train_gen.n // self.train_gen.batch_size
        self.step_size_val = self.val_gen.n // self.val_gen.batch_size

    def train(self, log_dir):
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        logdir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=10)

        self.model.fit(
            x=self.train_gen,
            steps_per_epoch=self.step_size_train,
            validation_data=self.val_gen,
            validation_steps=self.step_size_val,
            epochs=self.epoch,
            callbacks=[scheduler_callback, tensorboard_callback]
        )

    @staticmethod
    def scheduler(epoch, lr):
        if epoch < 15:
            return lr
        else:
            return lr * tf.math.exp(-0.1)