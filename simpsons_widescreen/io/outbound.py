import numpy as np
from skvideo.io import vwrite

from ..constants import EDGE_WIDTH


def write_video(video_path, predicted_targets, centre, edges):
    _write_video(video_path, "predicted_video", centre, predicted_targets)
    _write_video(video_path, "predicted_video", centre, edges)


def _write_video(video_path, file_name, centre, edges):
    left_edge = edges[:, :, :EDGE_WIDTH, :]
    right_edge = edges[:, :, -EDGE_WIDTH:, :]
    video_data = np.dstack([left_edge, centre, right_edge])

    vwrite(fname=video_path + f"{file_name}.mpg", videodata=video_data)
