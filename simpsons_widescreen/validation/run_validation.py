import logging

import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential

logging.basicConfig(level=logging.INFO)


def validate_trained_model(model: Sequential, testing_features: np.array, testing_targets: np.array, all_features: np.array) -> np.array:
    predicted_targets = predict(model, testing_features).astype(np.int16)

    r2 = r2_score(testing_targets.ravel(), predicted_targets.ravel())
    logging.info(f"R2 Score: {r2:.3}")

    return predict(model, all_features).astype(np.int16)


def predict(model: Sequential, video_array: np.array) -> np.array:
    normalized_images = (video_array - 127.5) / 127.5

    results = model.predict(normalized_images)

    return results