from typing import Tuple

import numpy as np

from ..constants import EDGE_WIDTH


def split_widescreen_frames(widescreen: np.array, prediction: bool = False) -> Tuple[np.array, np.array]:
    centre = widescreen[:, :, EDGE_WIDTH:-EDGE_WIDTH, :]

    if prediction:
        training = _grab_edges(widescreen)
        edges = np.zeros((centre.shape[0], centre.shape[1], 2 * EDGE_WIDTH, 3))
    else:
        training = _grab_edges(centre)
        edges = _grab_edges(widescreen)

    return training, edges, centre


def _grab_edges(widescreen: np.array) -> np.array:
    left_edge = widescreen[:, :, :EDGE_WIDTH, :]
    right_edge = widescreen[:, :, -EDGE_WIDTH:, :]

    return np.dstack([left_edge, right_edge])