from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K

from .model import build_model, build_perceptual_loss_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_model(
    training_features: np.array, 
    training_targets: np.array, 
    untrained_model: Optional[Model] = None,
    validation_data: Optional[Tuple[np.array, np.array]] = None,
    use_perceptual_loss: bool = True,
    epochs: int = 50,
    batch_size: int = 16,
    model_path: Optional[str] = None
) -> Model:
    """
    Train the model with improved loss functions and training strategies.
    
    Args:
        training_features: Input images (center frames)
        training_targets: Target edge images
        untrained_model: Optional pre-trained model for fine-tuning
        validation_data: Optional (val_features, val_targets) tuple
        use_perceptual_loss: Whether to use perceptual loss (recommended)
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained model
    """
    if untrained_model is None:
        if len(training_features.shape) == 4:
            input_size = (training_features.shape[1], training_features.shape[2])
        else:
            input_size = (720, 960)
        untrained_model = build_model(5, input_size=input_size)

    trained_model = _train_neural_network(
        untrained_model, 
        training_features, 
        training_targets,
        validation_data=validation_data,
        use_perceptual_loss=use_perceptual_loss,
        epochs=epochs,
        batch_size=batch_size,
        model_path=model_path
    )

    return trained_model


def _compute_ssim_loss(y_true, y_pred):
    """Compute SSIM loss for better structural similarity."""
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def _compute_perceptual_loss(vgg_model):
    """Create a perceptual loss function using VGG features."""
    def perceptual_loss(y_true, y_pred):
        y_true_scaled = (y_true + 1.0) * 127.5
        y_pred_scaled = (y_pred + 1.0) * 127.5
        
        target_size = (224, 224)
        y_true_resized = tf.image.resize(y_true_scaled, target_size, method='bilinear')
        y_pred_resized = tf.image.resize(y_pred_scaled, target_size, method='bilinear')
        
        y_true_vgg = tf.keras.applications.vgg19.preprocess_input(y_true_resized)
        y_pred_vgg = tf.keras.applications.vgg19.preprocess_input(y_pred_resized)
        
        true_features = vgg_model(y_true_vgg)
        pred_features = vgg_model(y_pred_vgg)
        
        loss = 0.0
        weights = [0.1, 0.1, 0.4, 0.4]
        
        for true_feat, pred_feat, weight in zip(true_features, pred_features, weights):
            loss += weight * tf.reduce_mean(tf.square(true_feat - pred_feat))
        
        return loss
    
    return perceptual_loss


def _combined_loss(vgg_model, mse_weight=0.1, ssim_weight=0.3, perceptual_weight=0.6):
    """Create a combined loss function."""
    mse = MeanSquaredError()
    perceptual = _compute_perceptual_loss(vgg_model)
    
    def loss(y_true, y_pred):
        mse_loss = mse(y_true, y_pred)
        ssim_loss = _compute_ssim_loss(y_true, y_pred)
        perc_loss = perceptual(y_true, y_pred)
        
        return mse_weight * mse_loss + ssim_weight * ssim_loss + perceptual_weight * perc_loss
    
    return loss


def _train_neural_network(
    model: Model, 
    training_features: np.array, 
    training_targets: np.array,
    validation_data: Optional[Tuple[np.array, np.array]] = None,
    use_perceptual_loss: bool = True,
    epochs: int = 50,
    batch_size: int = 16,
    model_path: Optional[str] = None
) -> Model:
    """
    Train the neural network with improved strategies.
    """
    normalized_features = (training_features.astype(np.float32) / 127.5) - 1.0
    normalized_targets = (training_targets.astype(np.float32) / 127.5) - 1.0
    
    val_features_norm = None
    val_targets_norm = None
    if validation_data is not None:
        val_features, val_targets = validation_data
        val_features_norm = (val_features.astype(np.float32) / 127.5) - 1.0
        val_targets_norm = (val_targets.astype(np.float32) / 127.5) - 1.0
    
    if use_perceptual_loss:
        vgg_model = build_perceptual_loss_model()
        loss_fn = _combined_loss(vgg_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
            loss=loss_fn
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
            loss=lambda y_true, y_pred: 0.7 * MeanSquaredError()(y_true, y_pred) + 0.3 * _compute_ssim_loss(y_true, y_pred)
        )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    
    checkpoint_path = None
    if validation_data and model_path:
        os.makedirs(model_path, exist_ok=True)
        checkpoint_path = os.path.join(model_path, 'best_model_checkpoint.h5')
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    history = model.fit(
        normalized_features,
        normalized_targets,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_features_norm, val_targets_norm) if validation_data else None,
        callbacks=callbacks,
        verbose=1
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    
    return model


def batch_generator(features: np.array, targets: np.array, train: bool = True, window_size: int = 5) -> Tuple[np.array, np.array]:
    """Generate batches with temporal windows (currently not used but kept for compatibility)."""
    batched_features, batched_targets = [], []
    for batch in range(features.shape[0] - window_size):
        batched_features.append(features[batch:batch + window_size])
        if train:
            batched_targets.append(targets[batch])

    return batched_features, batched_targets
