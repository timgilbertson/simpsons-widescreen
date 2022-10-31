from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, MaxPooling2D, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy

cross_entropy = BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

BUFFER_SIZE = 60000
BATCH_SIZE = 32
EPOCHS = 5

def make_generator_model(in_shape: Tuple[None, int, int, int] = (288, 128, 3)):
    """Build an image generating model."""
    model = Sequential([
        Conv2D(32,(3,3),activation="relu",input_shape=in_shape,padding='same'),
        Dropout(0.1),
        Conv2D(32,(3,3),activation="relu",input_shape=in_shape,padding='same'),
        MaxPooling2D((2,2)),

        Conv2D(64,  kernel_size = (3,3), activation='relu',padding='same'),
        Dropout(0.1),
        Conv2D(64,(3,3),activation="relu",input_shape=in_shape,padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(128,  kernel_size = (3,3), activation='relu',padding='same'),
        Dropout(0.1),
        Conv2D(128,(3,3),activation="relu",input_shape=in_shape,padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(256,  kernel_size = (3,3), activation='relu',padding='same'),
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

    return model


def make_discriminator_model():
    """Build a model to discriminate a real image from a generated image."""
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[BATCH_SIZE, 288, 128, 3]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


def _discriminator_loss(real_output, fake_output):
    """Calculate CE loss for the real and fake outputs."""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def _generator_loss(fake_output):
    """Calculate loss for the generator output."""
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images, targets, generator, discriminator):
    """Train the generator and discriminator models to generate realistic images."""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(images, training=True)
        
        real_output = discriminator(targets[None,:,:,:], training=True)
        fake_output = discriminator(generated_images[None,:,:,:], training=True)

        gen_loss = _generator_loss(fake_output)
        disc_loss = _discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def create_dataset(images, targets):
    """Create a TF dataset from an array of images."""
    return tf.data.Dataset.from_tensor_slices((images, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
