import os

import numpy as np
from skvideo.io import vreader


def read_video(video_path: str) -> np.array:
    frames = []
    for file_name in os.listdir(video_path):
        video = vreader(video_path + "/" + file_name, num_frames=1000, inputdict={"-r": 1})
        for frame in video:
            frames.append(frame)

    return np.asarray(frames)
