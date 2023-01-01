from simpsons_widescreen.widescreen import simpsons_widescreen


def main():
    params = {
        "input_prediction": "data/inputs/prediction_video/",
        "model_path": "data/outputs/trained_model",
        "output_prediction": "data/outputs/predicted_video/",
    }

    simpsons_widescreen(params)


if __name__ == "__main__":
    main()