import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ReLU, Softmax, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50


class FashionModel(object):
    def __init__(self):
        self.model = tf.keras.models.Model()

    def load_model(self, path):
        self.model = load_model(path)

    def create_model(self, input_shape=(224, 224, 3), num_classes=32, comp=True, init_lr=0.0002):
        self.model = self.__model_architecture_state_of_art(input_shape=input_shape, num_classes=num_classes)
        if comp:
            optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])


    @staticmethod
    def __model_architecture_state_of_art(input_shape, num_classes):
        resnet_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')
        for layer in resnet_model.layers[:-18]:
            layer.trainable = False
        dense_1 = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(resnet_model.output)
        dense_2_output = Dense(num_classes, activation='softmax', name='output_dense')(dense_1)

        return Model(inputs=resnet_model.input, outputs=dense_2_output, name='resnet_model')


if __name__ == '__main__':
    fm = FashionModel()
    fm.create_model()
    fm.model.summary()
