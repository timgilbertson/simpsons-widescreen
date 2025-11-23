from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19

from ..constants import EDGE_WIDTH


class EdgeExtractionLayer(layers.Layer):
    def __init__(self, edge_width=EDGE_WIDTH, **kwargs):
        super().__init__(**kwargs)
        self.edge_width = edge_width
    
    def call(self, inputs):
        left_edge = inputs[:, :, :self.edge_width, :]
        right_edge = inputs[:, :, -self.edge_width:, :]
        return tf.concat([left_edge, right_edge], axis=2)
    
    def get_config(self):
        config = super().get_config()
        config.update({"edge_width": self.edge_width})
        return config


def build_model(sequence_length: int, input_size: Tuple[int, int], output_size: Tuple[int, int] = (None, 720, 320, 3)) -> Model:
    """
    Build a U-Net style model with skip connections for edge generation.
    
    Args:
        sequence_length: Not used currently, kept for compatibility
        input_size: Input image dimensions (height, width)
        output_size: Output dimensions (not used, kept for compatibility)
    
    Returns:
        Compiled Keras model with perceptual loss support
    """
    in_shape = (input_size[0], input_size[1], 3)
    model = make_unet_model(in_shape)
    return model


def make_unet_model(input_shape: Tuple[int, int, int] = (288, 128, 3)) -> Model:
    """
    Create a U-Net architecture with skip connections for better edge generation.
    This architecture is more data-efficient than a simple encoder-decoder.
    """
    inputs = layers.Input(shape=input_shape)
    
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = layers.Dropout(0.1)(pool1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = layers.Dropout(0.1)(pool2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = layers.Dropout(0.2)(pool3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.2)(conv4)
    
    up5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
    up5 = layers.concatenate([up5, conv3], axis=3)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(0.2)(conv5)
    
    up6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    up6 = layers.concatenate([up6, conv2], axis=3)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Dropout(0.1)(conv6)
    
    up7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    up7 = layers.concatenate([up7, conv1], axis=3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    
    full_output = layers.Conv2D(3, 1, activation='tanh', kernel_initializer='he_normal')(conv7)
    outputs = EdgeExtractionLayer(edge_width=EDGE_WIDTH)(full_output)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse'
    )
    
    return model


def build_perceptual_loss_model():
    """
    Build a VGG19 model for perceptual loss computation.
    Uses intermediate layers to compute feature-based loss.
    """
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg.trainable = False
    
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4']
    outputs = [vgg.get_layer(name).output for name in layer_names]
    
    model = Model(inputs=vgg.input, outputs=outputs)
    return model