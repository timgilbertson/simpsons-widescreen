import logging
from typing import Dict

from .io.inbound import read_video
from .model.train import train_model
from .pre_process.split_frames import split_widescreen_frames
from .validation.create_validation import split_validation
from .validation.run_validation import validate_trained_model

logging.basicConfig(level=logging.INFO)


def simpsons_widescreen(params: Dict[str, str]):
    logging.info("Loading Episodes")
    widescreen_arrays = read_video(params["input_training"])

    logging.info("Splitting Widescreens")
    centre, edges = split_widescreen_frames(widescreen_arrays)

    logging.info("Splitting Training and Validation Sets")
    train_features, test_features, train_targets, test_targets = split_validation(centre, edges)

    logging.info("Training Neural Network")
    trained_model = train_model(train_features, train_targets)

    logging.info("Validating Trained Neural Network")
    validate_trained_model(trained_model, test_features, test_targets)
