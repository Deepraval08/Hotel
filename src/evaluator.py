# import mlflow
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from src.logger import logger

# class ModelEvaluator:

#     def evaluate(self, model, X_test, y_test):
#         preds = model.predict(X_test)

#         acc = accuracy_score(y_test, preds)
#         cm = confusion_matrix(y_test, preds)
#         report = classification_report(y_test, preds)

#         mlflow.log_metric("accuracy", acc)
#         mlflow.log_text(str(cm), "confusion_matrix.txt")
#         mlflow.log_text(report, "classification_report.txt")

#         logger.info(f"Accuracy: {acc}")
#         return acc
