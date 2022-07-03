from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, Conv3D
from keras.layers.pooling import MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam
import math

import keras.backend as K


def cross_entropy_error(y_true, y_pred):
    cse_eps = .000001
    y_pred = K.clip(y_pred, cse_eps, 1.0 - cse_eps)
    L = -K.sum(
        y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=1)
    cost = K.mean(L)
    return cost


# This should take no args, so that it has the full state of the network,
# making it easy to reload a model at a later point
def get_model():

    GRID_DIM = 40
    KERNEL_SIZE = 4
    LEARNING_RATE = 0.0001

    main_input = Input(
        shape=(GRID_DIM, GRID_DIM, GRID_DIM, 1),
        dtype='float32',
        name='main_input')

    # conv kernels
    x = Conv3D(
        kernel_initializer="he_normal",
        name="conv_3D_1",
        activation="relu",
        padding="valid",
        strides=[1, 1, 1],
        filters=64,
        kernel_size=(4, 4, 4))(main_input)
    x = MaxPooling3D(pool_size=(2, 2, 2), name='max_pool_1')(x)

    x = Conv3D(
        kernel_initializer="he_normal",
        name="conv_3D_2",
        activation="relu",
        padding="valid",
        strides=[1, 1, 1],
        filters=64,
        kernel_size=(4, 4, 4))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name='max_pool_2')(x)

    x = Conv3D(
        kernel_initializer="he_normal",
        name="conv_3D_3",
        activation="relu",
        padding="valid",
        strides=[1, 1, 1],
        filters=64,
        kernel_size=(4, 4, 4))(x)

    x = Flatten(name="flatten")(x)

    # dense compression kernels
    dim = ((
        (GRID_DIM - KERNEL_SIZE - KERNEL_SIZE + 2) / 2) - KERNEL_SIZE + 1) / 2
    size_1 = int(64 * dim * dim * dim / 2)
    x = Dense(
        size_1,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_compress_1")(x)

    size_2 = int(size_1 / 8)
    x = Dense(
        size_2,
        kernel_initializer='he_normal',
        activation='relu',
        name="dense_compress_2")(x)

    # dense reconstruction
    reconstruction = Dense(
        GRID_DIM * GRID_DIM * GRID_DIM,
        kernel_initializer='glorot_normal',
        activation='sigmoid',
        name='dense_reconstruct')(x)

    model = Model(inputs=main_input, outputs=reconstruction)

    # compile model
    print("Compiling model...")
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss=cross_entropy_error, optimizer=optimizer)

    return model
