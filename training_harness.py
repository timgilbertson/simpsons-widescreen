from simpsons_widescreen.widescreen import simpsons_widescreen


def main():
    params = {
        "input_training": "data/inputs/training_video",
        "input_prediction": "data/inputs/prediction_video/",
        "output_prediction": "data/outputs/",
        "model_path": "data/outputs/trained_model",
        "test_mode": True,
    }

    simpsons_widescreen(params)


if __name__ == "__main__":
    main()
