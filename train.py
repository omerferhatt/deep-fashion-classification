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
        logdir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau\
            (monitor='val_loss',
             patience=12,
             factor=0.5,
             verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch")

        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('models/model.h5')

        callbacks = [lr_reducer, tensorboard_callback, early_stopper, checkpoint]

        self.model.fit(
            x=self.train_gen,
            steps_per_epoch=200,
            validation_data=self.val_gen,
            validation_steps=200,
            epochs=self.epoch,
            callbacks=callbacks
        )
