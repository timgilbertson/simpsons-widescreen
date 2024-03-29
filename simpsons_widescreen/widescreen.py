import logging
import os
from typing import Dict

from tqdm import tqdm

from .io.inbound import read_video, read_episode
from .io.outbound import write_video, write_trained_model
from .model.train import train_model
from .pre_process.split_frames import split_widescreen_frames
from .validation.create_validation import split_validation
from .validation.run_validation import validate_trained_model

logging.basicConfig(level=logging.INFO)


def simpsons_widescreen_ext(params: Dict[str, str]):
    logging.info("Loading Episodes")
    widescreen_arrays = read_video(params["input_training"])

    logging.info("Splitting Widescreens")
    training, edges, centre = split_widescreen_frames(widescreen_arrays)

    logging.info("Splitting Training and Validation Sets")
    train_features, test_features, train_targets, test_targets = split_validation(training, edges)

    logging.info("Training Neural Network")
    trained_model = train_model(train_features, train_targets)

    logging.info("Validating Trained Neural Network")
    predicted_targets = validate_trained_model(trained_model, test_features, test_targets, training)

    logging.info("Writing Out Predicted Video")
    write_video(params["output_prediction"], predicted_targets, centre)


def simpsons_widescreen(params: Dict[str, str], model = None):
    file_list = os.listdir(params["input_prediction"])
    for file_name in file_list:
        prediction_video = read_episode(params["input_prediction"] + "/" + file_name, "384x288", 15)

    prediction, prediction_edges, _ = split_widescreen_frames(prediction_video, prediction=True)

    file_list = os.listdir(params["input_training"])
    count = 1
    for file_name in file_list:
        logging.info(f"Loading Episode {count} of {len(file_list)}")
        try:
            widescreen_arrays = read_episode(params["input_training"] + "/" + file_name, "512x288", 2)
        except:
            count += 1
            continue

        logging.info("Splitting Widescreens")
        training, edges, centre = split_widescreen_frames(widescreen_arrays)

        logging.info("Splitting Training and Validation Sets")
        train_features, test_features, train_targets, test_targets = split_validation(training, edges)

        if model:
            logging.info("Validating on Unseen Episode")
            validate_trained_model(model, test_features, test_targets, prediction)

        logging.info("Training Neural Network")
        model = train_model(train_features, train_targets, model)

        logging.info("Validating Trained Neural Network")
        predicted_targets = validate_trained_model(model, test_features, test_targets, prediction)

        logging.info("Writing Out Predicted Video")
        write_video(params["output_prediction"], predicted_targets, prediction_video, prediction_edges, "predicted_video")
        write_trained_model(params["output_prediction"], model)

        count += 1
