"""
Data augmentation utilities to improve data efficiency.
"""
from typing import Tuple

import numpy as np
import tensorflow as tf


def augment_data(features: np.array, targets: np.array, augment_prob: float = 0.5) -> Tuple[np.array, np.array]:
    """
    Apply data augmentation to training data.
    
    Args:
        features: Input images (center frames)
        targets: Target edge images
        augment_prob: Probability of applying each augmentation
    
    Returns:
        Augmented (features, targets) tuple
    """
    augmented_features = []
    augmented_targets = []
    
    for i in range(len(features)):
        feat = features[i].copy()
        targ = targets[i].copy()
        
        # Random horizontal flip (makes sense for left/right edges)
        if np.random.random() < augment_prob:
            feat = np.flip(feat, axis=1)
            # For edges, we need to swap left and right
            edge_width = targ.shape[1] // 2
            left_edge = targ[:, :edge_width, :]
            right_edge = targ[:, edge_width:, :]
            targ = np.concatenate([right_edge, left_edge], axis=1)
        
        # Random brightness adjustment
        if np.random.random() < augment_prob:
            brightness_factor = np.random.uniform(0.8, 1.2)
            feat = np.clip(feat * brightness_factor, 0, 255).astype(np.uint8)
            targ = np.clip(targ * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if np.random.random() < augment_prob:
            contrast_factor = np.random.uniform(0.9, 1.1)
            mean = feat.mean()
            feat = np.clip((feat - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
            mean = targ.mean()
            targ = np.clip((targ - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Random slight color shift (hue)
        if np.random.random() < augment_prob:
            shift = np.random.randint(-10, 10, size=3)
            feat = np.clip(feat.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            targ = np.clip(targ.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        
        augmented_features.append(feat)
        augmented_targets.append(targ)
    
    return np.array(augmented_features), np.array(augmented_targets)


def augment_batch(features: np.array, targets: np.array) -> Tuple[np.array, np.array]:
    """
    Apply augmentation to a batch of data (for use in training loop).
    This is more efficient than augmenting the entire dataset upfront.
    """
    batch_size = features.shape[0]
    augmented_features = []
    augmented_targets = []
    
    for i in range(batch_size):
        feat = features[i].copy()
        targ = targets[i].copy()
        
        # Random horizontal flip
        if np.random.random() < 0.5:
            feat = np.flip(feat, axis=1)
            edge_width = targ.shape[1] // 2
            left_edge = targ[:, :edge_width, :]
            right_edge = targ[:, edge_width:, :]
            targ = np.concatenate([right_edge, left_edge], axis=1)
        
        # Random brightness
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.85, 1.15)
            feat = np.clip(feat * brightness_factor, 0, 255).astype(np.uint8)
            targ = np.clip(targ * brightness_factor, 0, 255).astype(np.uint8)
        
        augmented_features.append(feat)
        augmented_targets.append(targ)
    
    return np.array(augmented_features), np.array(augmented_targets)

