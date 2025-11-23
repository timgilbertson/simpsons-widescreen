import os

import numpy as np
import imageio
import cv2
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm

from ..model.model import EdgeExtractionLayer


def read_video(video_path: str) -> np.array:
    frames = []
    for file_name in tqdm(os.listdir(video_path)):
        frame = read_episode(video_path + "/" + file_name)
        frames.append(frame)

    return np.asarray(frames)


def read_episode(file_name: str, input_shape: str, frame_rate: int = 1, full: bool = False, max_frames: int = None) -> np.array:
    frames = []
    if max_frames is not None:
        num_frames = max_frames
    else:
        num_frames = 1000 if not full else 5000
    
    width, height = map(int, input_shape.split('x'))
    
    reader = imageio.get_reader(file_name, 'ffmpeg')
    
    frame_skip = max(1, int(reader.get_meta_data()['fps'] / frame_rate)) if frame_rate > 0 else 1
    
    frame_count = 0
    for i, frame in enumerate(reader):
        if frame_count >= num_frames:
            break
        
        if i % frame_skip == 0:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
            frame_count += 1
    
    reader.close()
    return np.asarray(frames)


def read_model(model_path: str) -> Model:
    """
    Load a trained model. Handles models with custom loss functions by loading
    without compilation (not needed for inference).
    
    Args:
        model_path: Path to the saved model directory or file
    
    Returns:
        Loaded Keras model
    """
    if os.path.isdir(model_path):
        possible_files = [
            os.path.join(model_path, "trained_model.keras"),
        ]
        for path in possible_files:
            if os.path.exists(path):
                model_path = path
                break
        if not any(os.path.exists(p) for p in possible_files):
            pass

    custom_objects = {
        'EdgeExtractionLayer': EdgeExtractionLayer,
    }
    
    model = load_model(model_path, compile=False, custom_objects=custom_objects)

    return model

