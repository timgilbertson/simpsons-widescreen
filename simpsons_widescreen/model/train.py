from typing import Tuple

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.data import Dataset
from tqdm import tqdm

from .model import train_step, EPOCHS, create_dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_model(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    generator: Sequential,
    discriminator: Sequential
) -> Sequential:
    """Train a GAN"""
    scaled_images = _scale_images(training_features)
    dataset = create_dataset(scaled_images, training_targets)

    for _ in tqdm(range(EPOCHS)):
        for image_batch in dataset:
            features = image_batch[0]
            targets = image_batch[1]
            train_step(features, targets, generator, discriminator)

    return generator, discriminator


def _scale_images(images: np.ndarray) -> np.ndarray:
    """Simple pixel value scaler between -1 and 1"""
    return (images - 127.5) / 127.5
