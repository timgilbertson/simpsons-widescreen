from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Dropout, Activation, MaxPooling3D, LSTM, Reshape


def build_model(sequence_length: int, input_size: Tuple[int, int], output_size: Tuple[int, int] = (720, 320)):
    in_shape = (sequence_length, input_size[0], input_size[1], 3)
    model = Sequential()
    model.add(ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True, input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(ConvLSTM2D(64, kernel_size=(5, 5), padding='valid', return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(Activation('relu'))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(Activation('relu'))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dense(320))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Reshape((sequence_length, output_size[0] * output_size[1] * 3)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])