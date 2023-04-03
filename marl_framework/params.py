import logging
import os
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


def load_params(config_file_path: str) -> Dict:
    logger.info(
        "\n----------------------------------- USED HYPERPARAMETERS -----------------------------------\n"
    )

    if not os.path.isfile(config_file_path):
        logger.error(f"Config file {config_file_path} does not exist!")
        raise FileNotFoundError

    try:
        with open(config_file_path, "rb") as config_file:
            params = yaml.load(config_file.read(), Loader=yaml.Loader)
        logger.info(params)
        return params
    except Exception as e:
        logger.error(f"Error while reading config file! {e}")
        raise Exception(e)
