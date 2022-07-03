from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def split_validation(centre: np.array, edges: np.array, test_split: float = 0.15) -> Tuple[np.array, np.array, np.array, np.array]:
    return train_test_split(centre, edges, test_size=test_split, random_state=97)
