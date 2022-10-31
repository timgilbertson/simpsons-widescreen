import os

import numpy as np
from skvideo.io import vreader
from tqdm import tqdm


def read_video(video_path: str) -> np.array:
    frames = []
    for file_name in tqdm(os.listdir(video_path)):
        frame = read_episode(video_path + "/" + file_name)
        frames.append(frame)

    return np.asarray(frames)


def read_episode(file_name: str, input_shape: str, frame_rate: int = 15) -> np.array:
    frames = []
    video = vreader(file_name, num_frames=2000, outputdict={"-r": str(frame_rate), "-s": input_shape})
    for frame in video:
        frames.append(frame)

    return np.asarray(frames)