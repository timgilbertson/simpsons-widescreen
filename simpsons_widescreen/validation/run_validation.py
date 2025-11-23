import logging

import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model, Sequential

logging.basicConfig(level=logging.INFO)


def validate_trained_model(model: Model, testing_features: np.array, testing_targets: np.array, all_features: np.array) -> np.array:
    predicted_targets = predict(model, testing_features).astype(np.int16)

    r2 = r2_score(testing_targets.ravel(), predicted_targets.ravel())
    logging.info(f"R2 Score: {r2:.3}")

    return predict(model, all_features).astype(np.int16)


def predict(model: Model, video_array: np.array) -> np.array:
    """
    Predict edge pixels from center frames.
    
    Args:
        model: Trained Keras model
        video_array: Input video frames (center frames) in [0, 255] range
    
    Returns:
        Predicted edge pixels in [0, 255] range
    """
    normalized_images = (video_array.astype(np.float32) / 127.5) - 1.0

    results = model.predict(normalized_images, verbose=0)
    
    results = ((results + 1.0) * 127.5).clip(0, 255)

    return results