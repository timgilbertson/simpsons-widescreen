from typing import Dict
import os

import numpy as np

from .io.inbound import read_episode, read_model
from .io.outbound import write_video
from .pre_process.split_frames import split_widescreen_frames
from .validation.run_validation import predict


def simpsons_widescreen(params: Dict[str, str]):
    file_list = os.listdir(params["input_prediction"])
    for file_name in file_list:
        prediction_video = read_episode(params["input_prediction"] + "/" + file_name, "384x288", 30, full=True)

        prediction, prediction_edges, _ = split_widescreen_frames(prediction_video, prediction=True)

        model = read_model(params["model_path"])

        widescreened = predict(model, prediction).astype(np.int16)

        write_video(params["output_prediction"], widescreened, prediction_video, prediction_edges, file_name)
