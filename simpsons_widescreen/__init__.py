import argparse

from .widescreen import simpsons_widescreen


def main():
    parser = argparse.ArgumentParser(description="Simpsons Widescreen Predictor")
    parser.add_argument("--input-training", help="Widescreen training video location", required=True)
    parser.add_argument("--input-prediction", help="Input 4x3 prediction video location", required=True)
    parser.add_argument("--output-prediction", help="Output 4x3 prediction video location", required=True)

    args = parser.parse_args()
    params = {arg: getattr(args, arg) for arg in vars(args)}
    simpsons_widescreen(params)
