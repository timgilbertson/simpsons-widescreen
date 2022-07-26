import os

import numpy as np
from skvideo.io import vreader
from tqdm import tqdm


def read_video(video_path: str) -> np.array:
    frames = []
    for file_name in tqdm(os.listdir(video_path)[:10]):
        video = vreader(video_path + "/" + file_name, num_frames=100, outputdict={"-r": "1"})
        for frame in video:
            frames.append(frame)

    return np.asarray(frames)
