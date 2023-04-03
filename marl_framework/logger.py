import logging
import os
import time

import constants


def setup_logger() -> logging.Logger:
    os.makedirs(constants.LOG_DIR, exist_ok=True)
    logger = logging.getLogger(__package__)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = []

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console.setFormatter(console_formatter)

    logger.addHandler(console)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    log_filename = f"log_{timestamp}.log"
    log_file_path = os.path.join(constants.LOG_DIR, log_filename)

    file_handler = logging.FileHandler(
        filename=log_file_path, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    return logger
