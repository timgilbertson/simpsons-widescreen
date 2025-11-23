from simpsons_widescreen.generate_widescreen import simpsons_widescreen


def main():
    params = {
        "input_prediction": "data/inputs/prediction_video/",
        "model_path": "data/outputs/trained_model/trained_model.keras",
        "output_prediction": "data/outputs/predicted_video/",
        "test_mode": True,
    }

    simpsons_widescreen(params)


if __name__ == "__main__":
    main()