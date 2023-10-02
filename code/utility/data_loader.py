import logging

import pandas as pd

from code.utility.path_utils import get_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = get_data_path("BodyFat.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Data is empty or not loaded properly.")
    logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.\n")

    return df
