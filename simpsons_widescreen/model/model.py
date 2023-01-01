from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPooling2D


def build_model(sequence_length: int, input_size: Tuple[int, int], output_size: Tuple[int, int] = (None, 720, 320, 3)) -> Sequential:
    in_shape = (sequence_length, input_size[0], input_size[1], 3)
    
    model = make_generator_model()

    return model


def make_generator_model(in_shape: Tuple[None, int, int, int] = (288, 128, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=in_shape, padding='same'),
        Dropout(0.1),
        Conv2D(32, (3, 3), activation="relu", input_shape=in_shape, padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(64,  kernel_size = (3, 3), activation='relu', padding='same'),
        Dropout(0.1),
        Conv2D(64, (3, 3), activation="relu", input_shape=in_shape, padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(128,  kernel_size = (3, 3), activation='relu', padding='same'),
        Dropout(0.1),
        Conv2D(128, (3, 3), activation="relu", input_shape=in_shape, padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(256,  kernel_size = (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
    
        Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),

        Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
    
        Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        Dropout(0.2),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),

        Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same'),
        Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        Dropout(0.2),
        Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),

        Conv2D(3, (1, 1), activation='linear')
    ])

    model.compile(optimizer="adam", loss="mse")

    return model