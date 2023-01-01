import os

import numpy as np
from skvideo.io import vreader
from tensorflow.keras.models import Sequential, load_model
from tqdm import tqdm


def read_video(video_path: str) -> np.array:
    frames = []
    for file_name in tqdm(os.listdir(video_path)):
        frame = read_episode(video_path + "/" + file_name)
        frames.append(frame)

    return np.asarray(frames)


def read_episode(file_name: str, input_shape: str, frame_rate: int = 1, full: bool = False) -> np.array:
    frames = []
    num_frames = 1000 if not full else None
    video = vreader(file_name, num_frames=num_frames, outputdict={"-r": str(frame_rate), "-s": input_shape})
    for frame in video:
        frames.append(frame)

    return np.asarray(frames)


def read_model(model_path: str) -> Sequential:
    return load_model(model_path)
