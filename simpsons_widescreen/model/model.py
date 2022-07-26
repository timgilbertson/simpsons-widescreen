from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose


def build_model(sequence_length: int, input_size: Tuple[int, int], output_size: Tuple[int, int] = (None, 720, 320, 3)) -> Sequential:
    in_shape = (sequence_length, input_size[0], input_size[1], 3)
    
    model = make_generator_model()

    return model


def make_generator_model(output_shape: Tuple[None, int, int, int] = (720, 320, 3)):
    model = Sequential()
    model.add(Dense(3, use_bias=False, input_shape=(output_shape)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # model.add(Reshape((180, 80, 32)))
    # assert model.output_shape == (None, 180, 80, 32)  # Note: None is the batch size

    model.add(Conv2DTranspose(16, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # assert model.output_shape == (None, 180, 80, 16)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 360, 160, 8)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='linear'))

    model.compile(optimizer="adam", loss="mse")

    return model