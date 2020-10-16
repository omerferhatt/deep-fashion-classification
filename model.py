import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ReLU, Softmax, BatchNormalization, Dropout


class FashionModel(object):
    def __init__(self):
        self.model = tf.keras.models.Model()

    def load_model(self, path):
        self.model = load_model(path)

    def create_model(self, input_shape=(192, 192, 3), num_classes=32, comp=True, init_lr=0.001):
        self.model = self.__model_architecture(input_shape=input_shape, num_classes=num_classes)
        if comp:
            optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    @staticmethod
    def __model_architecture(input_shape, num_classes):
        inp_layer = Input(shape=input_shape, name='input_layer')

        conv1_1 = Conv2D(32, kernel_size=(5, 5), padding='valid', name='conv1_1')(inp_layer)
        b_norm1_1 = BatchNormalization(name='b_norm1_1')(conv1_1)
        act1_1 = ReLU(name='act1_1')(b_norm1_1)

        conv1_2 = Conv2D(64, kernel_size=(5, 5), padding='valid', name='conv1_2')(act1_1)
        b_norm1_2 = BatchNormalization(name='b_norm1_2')(conv1_2)
        act1_2 = ReLU(name='act1_2')(b_norm1_2)

        pool1 = MaxPool2D(pool_size=(2, 2), name='pool1')(act1_2)

        conv2_1 = Conv2D(64, kernel_size=(4, 4), padding='valid', name='conv2_1')(pool1)
        b_norm2_1 = BatchNormalization(name='b_norm2_1')(conv2_1)
        act2_1 = ReLU(name='act2_1')(b_norm2_1)

        conv2_2 = Conv2D(64, kernel_size=(4, 4), padding='valid', name='conv2_2')(act2_1)
        b_norm2_2 = BatchNormalization(name='b_norm2_2')(conv2_2)
        act2_2 = ReLU(name='act2_2')(b_norm2_2)

        pool2 = MaxPool2D(pool_size=(2, 2), name='pool2')(act2_2)

        conv3_1 = Conv2D(128, kernel_size=(3, 3), padding='valid', name='conv3_1')(pool2)
        b_norm3_1 = BatchNormalization(name='b_norm3_1')(conv3_1)
        act3_1 = ReLU(name='act3_1')(b_norm3_1)

        conv3_2 = Conv2D(128, kernel_size=(3, 3), padding='valid', name='conv3_2')(act3_1)
        b_norm3_2 = BatchNormalization(name='b_norm3_2')(conv3_2)
        act3_2 = ReLU(name='act3_2')(b_norm3_2)

        conv3_3 = Conv2D(128, kernel_size=(3, 3), padding='valid', name='conv3_3')(act3_2)
        b_norm3_3 = BatchNormalization(name='b_norm3_3')(conv3_3)
        act3_3 = ReLU(name='act3_3')(b_norm3_3)

        pool3 = MaxPool2D(pool_size=(2, 2), name='pool3')(act3_3)

        conv4_1 = Conv2D(256, kernel_size=(3, 3), padding='valid', name='conv4_1')(pool3)
        b_norm4_1 = BatchNormalization(name='b_norm4_1')(conv4_1)
        act4_1 = ReLU(name='act4_1')(b_norm4_1)

        conv4_2 = Conv2D(256, kernel_size=(3, 3), padding='valid', name='conv4_2')(act4_1)
        b_norm4_2 = BatchNormalization(name='b_norm4_2')(conv4_2)
        act4_2 = ReLU(name='act4_2')(b_norm4_2)

        conv4_3 = Conv2D(512, kernel_size=(3, 3), padding='valid', name='conv4_3')(act4_2)
        b_norm4_3 = BatchNormalization(name='b_norm4_3')(conv4_3)
        act4_3 = ReLU(name='act4_3')(b_norm4_3)

        gap = GlobalAveragePooling2D(name='gap')(act4_3)
        drop1 = Dropout(0.3, name='drop1')(gap)

        dense1 = Dense(512, name='dense1')(drop1)
        act4 = ReLU(name='act4')(dense1)
        drop2 = Dropout(0.3, name='drop2')(act4)

        dense2 = Dense(256, name='dense2')(drop2)
        act5 = ReLU(name='act5')(dense2)
        drop3 = Dropout(0.3, name='drop3')(act5)

        dense3 = Dense(num_classes, name='dense3')(drop3)
        out_act6 = Softmax(name='output_act6')(dense3)

        return Model(inputs=inp_layer, outputs=out_act6)


if __name__ == '__main__':
    fm = FashionModel()
    fm.create_model()
    fm.model.summary()
