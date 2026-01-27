from src.pipeline import MLPipeline
import mlflow.sklearn
from src.logger import logger
import joblib
import os


if __name__ == "__main__":

    pipeline = MLPipeline(
        data_path="data/hotel.csv",
        target_column="booking_status"
    )

    results = pipeline.run()

if len(results) > 0:
    best_model = results["random_forest"]
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/hotel_model.pkl")


       
 