import mlflow
import mlflow.sklearn
import os, sys, random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.logger import logger
from src.exception import CustomException


class ModelTrainer:

    def __init__(self):

        self.models = {
            "logistic": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(random_state=42),
            "xgboost": XGBClassifier(
                eval_metric="logloss",
                tree_method="hist",
                random_state=42
            )
        }

        self.param_space = {

            "logistic": {
                "C": [0.01, 0.1, 1, 10]
            },

            "random_forest": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5]
            },

            "xgboost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "min_child_weight": [3, 5]
            }
        }

    def _sample_params(self, space):
        return {k: random.choice(v) for k, v in space.items()}

    def train(self, pipeline, X_train, y_train, X_test, y_test):

        try:
                      
            reports_dir = "artifacts/reports"
            # If folder exists, delete it completely
            if os.path.exists(reports_dir):
                    shutil.rmtree(reports_dir)
                    # Recreate empty folder
            os.makedirs(reports_dir, exist_ok=True)

            mlflow.set_experiment("hotel_reservation_v2")

            best_models = {}

            for name, model in self.models.items():

                logger.info(f"Training {name}")

                best_score = -1
                best_estimator = None
                best_params = None

                # ---------------- RANDOM SEARCH ----------------
                for i in range(10):

                    params = self._sample_params(self.param_space[name])

                    pipeline.set_params(
                        model=model,
                        **{f"model__{k}": v for k, v in params.items()}
                    )

                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)

                    acc = accuracy_score(y_test, preds)

                    if acc > best_score:
                        best_score = acc
                        best_estimator = copy.deepcopy(pipeline)
                        best_params = params

                # ---------------- LOG ONLY BEST MODEL ----------------
                with mlflow.start_run(run_name=name):

                    preds = best_estimator.predict(X_test)

                    acc = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds)
                    rec = recall_score(y_test, preds)
                    f1 = f1_score(y_test, preds)

                    mlflow.log_params(best_params)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("precision", prec)
                    mlflow.log_metric("recall", rec)
                    mlflow.log_metric("f1", f1)

                    cm = confusion_matrix(y_test, preds)
                    report = classification_report(y_test, preds)

                    os.makedirs("artifacts/reports", exist_ok=True)

                    cm_path = f"artifacts/reports/{name}_cm.png"
                    rpt_path = f"artifacts/reports/{name}_report.txt"
                    xticks = ["Not_Canceled (0)", "Canceled (1)"]
                    yticks = ["Not_Canceled (0)", "Canceled (1)"]


                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=xticks, yticklabels=yticks)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    plt.savefig(cm_path, bbox_inches='tight')
                    plt.close()

                    with open(rpt_path, "w") as f:
                        f.write(report)

                    mlflow.log_artifact(cm_path)
                    mlflow.log_artifact(rpt_path)

                    mlflow.sklearn.log_model(
                        best_estimator,
                        name="model"
                    )

                    best_models[name] = best_estimator

            return best_models

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)
