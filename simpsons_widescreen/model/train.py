from typing import Tuple

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.data import Dataset

from .model import build_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_model(training_features: np.array, training_targets: np.array, untrained_model: Sequential = None) -> Sequential:
    if not untrained_model:
        untrained_model = build_model(5, input_size=(720, 960))

    trained_model = _train_neural_network(untrained_model, training_features, training_targets)

    return trained_model
    

def _train_neural_network(model: Sequential, training_features: np.array, training_targets: np.array) -> Sequential:
    normalized_images = (training_features - 127.5) / 127.5

    model.fit(normalized_images, training_targets, epochs=15)

    return model


def batch_generator(features: np.array, targets: np.array, train: bool = True, window_size: int = 5) -> Tuple[np.array, np.array]:
    batched_features, batched_targets = [], []
    for batch in range(features.shape[0] - window_size):
        batched_features.append(features[batch:batch + window_size])
        if train:
            batched_targets.append(targets[batch])

    return batched_features, batched_targets
