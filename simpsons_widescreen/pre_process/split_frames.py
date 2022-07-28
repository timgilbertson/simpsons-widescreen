from typing import Tuple

import numpy as np

from ..constants import FOUR_THREE_WIDTH, EDGE_WIDTH


def split_widescreen_frames(widescreen: np.array) -> Tuple[np.array, np.array]:
    centre = widescreen[:, :, EDGE_WIDTH:-EDGE_WIDTH, :]

    training = _grab_edges(centre)
    edges = _grab_edges(widescreen)

    return training, edges, centre


def _grab_edges(widescreen: np.array) -> np.array:
    left_edge = widescreen[:, :, :EDGE_WIDTH, :]
    right_edge = widescreen[:, :, -EDGE_WIDTH:, :]

    return np.dstack([left_edge, right_edge])