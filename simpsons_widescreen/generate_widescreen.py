from typing import Dict
import os

import numpy as np
from tqdm import tqdm

from .io.inbound import read_episode, read_model
from .io.outbound import write_video
from .validation.run_validation import predict
from .constants import EDGE_WIDTH


def simpsons_widescreen(params: Dict[str, str]):
    test_mode = params.get("test_mode", False)
    
    model = read_model(params["model_path"])
    
    file_list = os.listdir(params["input_prediction"])
    for file_name in tqdm(file_list):
        max_frames = 300 if test_mode else None
        prediction_video = read_episode(params["input_prediction"] + "/" + file_name, "384x288", 30, full=not test_mode, max_frames=max_frames)

        prediction_centre = prediction_video
        prediction_edges = np.zeros((prediction_centre.shape[0], prediction_centre.shape[1], 2 * EDGE_WIDTH, 3))

        widescreened = predict(model, prediction_centre).astype(np.int16)

        if params["output_prediction"].endswith('.mp4'):
            output_path = params["output_prediction"]
        else:
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(params["output_prediction"], f"{base_name}.mp4")
        write_video(output_path, widescreened, prediction_video, prediction_edges, file_name)
