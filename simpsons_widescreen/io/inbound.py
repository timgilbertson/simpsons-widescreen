import pdb
import numpy as np
from skvideo.io import vreader


def read_video(video_path: str) -> np.array:
    frames = []
    video = vreader(video_path)
    for frame in video:
        frames.append(frame)
    import pdb; pdb.set_trace()
