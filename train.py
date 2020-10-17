import tensorflow as tf
import os
import datetime


class Trainer(object):
    def __init__(self, model: tf.keras.models.Model, train_gen: tf.keras.preprocessing.image.ImageDataGenerator,
                 val_gen: tf.keras.preprocessing.image.ImageDataGenerator, epoch: int, step: int):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.epoch = epoch
        self.step = step

        self.step_size_train = self.train_gen.n // self.train_gen.batch_size
        self.step_size_val = self.val_gen.n // self.val_gen.batch_size

    def train(self, log_dir):
        logdir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau\
            (monitor='val_loss',
             patience=10,
             factor=0.7,
             verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch")

        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            verbose=1)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/model.h5',
            monitor='val_accuracy',
            save_weights_only=True,
            mode='max',
            save_best_only=True)

        callbacks = [lr_reducer, tensorboard_callback, early_stopper, checkpoint]
        if self.step != 0:
            self.model.fit(
                x=self.train_gen,
                steps_per_epoch=self.step,
                validation_data=self.val_gen,
                validation_steps=self.step,
                epochs=self.epoch,
                callbacks=callbacks
            )
        else:
            self.model.fit(
                x=self.train_gen,
                steps_per_epoch=self.step_size_train,
                validation_data=self.val_gen,
                validation_steps=self.step_size_val,
                epochs=self.epoch,
                callbacks=callbacks
            )
