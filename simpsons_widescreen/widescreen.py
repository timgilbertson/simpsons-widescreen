import logging
from typing import Dict

from .io.inbound import read_video

logging.basicConfig(level=logging.INFO)


def simpsons_widescreen(params: Dict[str, str]):
    logging.info("Loading Episodes")
    read_video(params["input_training"])
