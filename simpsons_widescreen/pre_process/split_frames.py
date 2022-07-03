from typing import Tuple

import numpy as np

from ..constants import FOUR_THREE_WIDTH, EDGE_WIDTH


def split_widescreen_frames(widescreen: np.array) -> Tuple[np.array, np.array]:
    centre = widescreen[:, :, EDGE_WIDTH:-EDGE_WIDTH, :]
    left_edge = widescreen[:, :, :EDGE_WIDTH, :]
    right_edge = widescreen[:, :, -EDGE_WIDTH:, :]

    return centre, np.dstack([left_edge, right_edge])
