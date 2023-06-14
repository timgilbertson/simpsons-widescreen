import numpy as np
from skvideo.io import vwrite
from tensorflow.keras.models import Sequential

from ..constants import EDGE_WIDTH


def write_video(video_path, predicted_targets, centre, edges, file_name):
    _write_video(video_path, file_name, centre, predicted_targets)


def _write_video(video_path, file_name, centre, edges):
    left_edge = edges[:, :, :EDGE_WIDTH, :]
    right_edge = edges[:, :, -EDGE_WIDTH:, :]
    video_data = np.dstack([left_edge, centre, right_edge]).astype(np.int16)

    vwrite(fname=video_path + f"{file_name}.mpg", videodata=video_data, outputdict={'-b': '30000000'})


def write_trained_model(model_path: str, trained_model: Sequential):
    trained_model.save(model_path + "/trained_model")