import sys
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessor
from src.trainer import ModelTrainer
from src.logger import logger
from src.exception import CustomException


class MLPipeline:

    def __init__(self, data_path, target_column):

        self.ingestion = DataIngestion(data_path)
        self.preprocessor = DataPreprocessor(target_column)
        self.trainer = ModelTrainer()

        logger.info("MLPipeline initialized successfully")

    def run(self):

        try:
            logger.info("========== ML Pipeline Started ==========")

            # Load data
            df = self.ingestion.load_data()

            # Split features/target
            X, y = self.preprocessor.split_features(df)
            
            if "Booking_ID" in X.columns:
                X = X.drop("Booking_ID", axis=1)

            y = self.preprocessor.encode_target(y)

            preprocessor_obj = self.preprocessor.get_preprocessor(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # Base pipeline (model set inside trainer)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_obj),
                    ("smote", SMOTE(random_state=42)),
                    ("model", None)
                ]
            )

            # smote = SMOTE(random_state=42)
            # X_train, y_train = smote.fit_resample(X_train, y_train)

            # Trainer handles everything
            best_models = self.trainer.train(
                pipeline,
                X_train,
                y_train,
                X_test,
                y_test
            )

            logger.info("========== ML Pipeline Finished ==========")

            return best_models

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)
