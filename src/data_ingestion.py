import pandas as pd
import os, sys
from src.logger import logger
from src.exception import CustomException


class DataIngestion:
    def __init__(self, path):
        self.path = path


    def load_data(self):
        try:
            if not os.path.exists(self.path):
                raise FileNotFoundError("Dataset file not found")
            df = pd.read_csv(self.path)
            logger.info("Dataset loaded successfully")
            return df
        except Exception as e:
            custom_error = CustomException(e, sys)
            logger.exception(custom_error)   # logs full traceback
            raise custom_error