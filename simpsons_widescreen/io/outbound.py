import os

import numpy as np
import imageio
from tensorflow.keras.models import Sequential

from ..constants import EDGE_WIDTH


def write_video(video_path, predicted_targets, centre, edges, file_name):
    _write_video(video_path, file_name, centre, predicted_targets)


def _write_video(video_path, file_name, centre, edges):
    """
    Write video using imageio.
    
    Args:
        video_path: Directory to save video or full file path
        file_name: Name of output file (without extension)
        centre: Center frames array
        edges: Edge frames array
    """
    left_edge = edges[:, :, :EDGE_WIDTH, :]
    right_edge = edges[:, :, -EDGE_WIDTH:, :]

    video_data = np.concatenate([left_edge, centre, right_edge], axis=2)
    
    video_data = np.clip(video_data, 0, 255).astype(np.uint8)
    
    if video_path.endswith('.mp4'):
        output_path = video_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        os.makedirs(video_path, exist_ok=True)
        output_path = os.path.join(video_path, f"{file_name}.mp4")
    
    writer = imageio.get_writer(output_path, fps=30, codec='libx264', bitrate='30M', quality=8)
    
    for frame in video_data:
        writer.append_data(frame)
    
    writer.close()


def write_trained_model(model_path: str, trained_model: Sequential):
    """
    Save a trained model to disk.
    
    Args:
        model_path: Directory path where model should be saved
        trained_model: The trained Keras model to save
    """
    trained_model_dir = os.path.join(model_path, "trained_model")
    os.makedirs(trained_model_dir, exist_ok=True)
    
    model_file = os.path.join(trained_model_dir, "trained_model.keras")
    trained_model.save(model_file)