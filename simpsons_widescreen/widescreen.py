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
    test_mode = params.get("test_mode", False)
    
    logging.info("Loading Episodes")
    widescreen_arrays = read_video(params["input_training"])

    logging.info("Splitting Widescreens")
    training, edges, centre = split_widescreen_frames(widescreen_arrays)

    if test_mode:
        centre = centre[:100]
        edges = edges[:100]

    logging.info("Splitting Training and Validation Sets")
    train_features, test_features, train_targets, test_targets = split_validation(centre, edges)

    logging.info("Training Neural Network")
    model_path = params.get("model_path", "data/outputs/trained_model")
    
    trained_model = train_model(
        train_features, 
        train_targets,
        validation_data=(test_features, test_targets),
        use_perceptual_loss=not test_mode,
        epochs=1 if test_mode else 5,
        batch_size=4 if test_mode else 16,
        model_path=model_path
    )

    logging.info("Validating Trained Neural Network")
    predicted_targets = validate_trained_model(trained_model, test_features, test_targets, centre)

    logging.info("Writing Out Predicted Video")
    if params["output_prediction"].endswith('.mp4'):
        output_path = params["output_prediction"]
    else:
        output_path = os.path.join(params["output_prediction"], "predicted_video.mp4")
    write_video(output_path, predicted_targets, centre, edges, "predicted_video")


def simpsons_widescreen(params: Dict[str, str], model = None):
    test_mode = params.get("test_mode", False)
    
    file_list = os.listdir(params["input_prediction"])
    for file_name in file_list:
        max_frames = 50 if test_mode else None
        prediction_video = read_episode(params["input_prediction"] + "/" + file_name, "384x288", 15, max_frames=max_frames)

    _, prediction_edges, prediction_centre = split_widescreen_frames(prediction_video, prediction=True)

    file_list = os.listdir(params["input_training"])
    if test_mode:
        file_list = file_list[:1]
    
    count = 1
    for file_name in file_list:
        logging.info(f"Loading Episode {count} of {len(file_list)}")
        try:
            max_frames = 50 if test_mode else None
            widescreen_arrays = read_episode(params["input_training"] + "/" + file_name, "512x288", 2, max_frames=max_frames)
        except:
            count += 1
            continue

        logging.info("Splitting Widescreens")
        training, edges, centre = split_widescreen_frames(widescreen_arrays)

        logging.info("Splitting Training and Validation Sets")
        train_features, test_features, train_targets, test_targets = split_validation(centre, edges)

        if model:
            logging.info("Validating on Unseen Episode")
            validate_trained_model(model, test_features, test_targets, prediction_centre)

        logging.info("Training Neural Network")
        model_path = params.get("model_path", "data/outputs/trained_model")
        
        model = train_model(
            train_features, 
            train_targets, 
            untrained_model=model,
            validation_data=(test_features, test_targets),
            use_perceptual_loss=not test_mode,
            epochs=1 if test_mode else 5,
            batch_size=4 if test_mode else 16,
            model_path=model_path
        )

        logging.info("Validating Trained Neural Network")
        predicted_targets = validate_trained_model(model, test_features, test_targets, prediction_centre)

        logging.info("Writing Out Predicted Video")
        if params["output_prediction"].endswith('.mp4'):
            output_path = params["output_prediction"]
        else:
            output_path = os.path.join(params["output_prediction"], "predicted_video.mp4")
        write_video(output_path, predicted_targets, prediction_video, prediction_edges, "predicted_video")
        write_trained_model(params["output_prediction"], model)

        count += 1
