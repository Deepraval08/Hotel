import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logger
from src.exception import CustomException
import logging


class DataPreprocessor:

    def __init__(self, target_col):
        self.target_col = target_col

    def split_features(self, df):
        try:
            X = df.drop(self.target_col, axis=1)
            y = df[self.target_col]
            return X, y
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def encode_target(self, y):
        """
        Encode target variable manually using a dictionary mapping.
        
        Parameters:
            y (pd.Series): Target column with categorical labels.
        
        Returns:
            pd.Series: Encoded target with 0 and 1
        """
        mapping = {'Not_Canceled': 0, 'Canceled': 1}
        return y.map(mapping)

    def get_preprocessor(self, X):

        num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        
        no_scale_cols = ["required_car_parking_space", "repeated_guest"]
        scale_cols = [col for col in num_cols if col not in no_scale_cols]
        passthrough_cols = [col for col in num_cols if col in no_scale_cols]

        logging.info(f"Numerical columns: {list(num_cols)}")
        logging.info(f"Categorical columns: {list(cat_cols)}")

        num_scaled_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ])

        num_no_scale_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
           ("onehot", OneHotEncoder(
                handle_unknown="ignore",
            ))
        ])

        preprocessor = ColumnTransformer([
        ("num_scaled", num_scaled_pipeline, scale_cols),
        ("num_no_scale", num_no_scale_pipeline, passthrough_cols),
        ("cat", cat_pipeline, cat_cols)
        ])

        return preprocessor