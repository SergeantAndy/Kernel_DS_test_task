from src.service import Service
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
from typing import Tuple
import pandas as pd

class Modeling:
    def __init__(self, service_instance: Service):
        """
        Initializes the Modeling instance.

        Parameters:
        - service_instance (Service): An instance of the Service class.
        """
        self.service = service_instance
        self.config = self.service.get_config()
        self.model = None  # Placeholder for the trained model

    def _prepare_data(self, dataframe, target_variable: str, ignore_columns: list) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares the data by extracting features (X) and target variable (y).

        Parameters:
        - dataframe (pd.DataFrame): Input DataFrame.
        - target_variable (str): Name of the target variable column.
        - ignore_columns (list): List of columns to be ignored.

        Returns:
        - Tuple[pd.DataFrame, pd.Series]: Features (X) and target variable (y).
        """
        X = dataframe.drop(ignore_columns + [target_variable], axis=1)
        y = dataframe[target_variable]

        return X, y

    @Service.log_output
    def fit_model(self, train_data: pd.DataFrame):
        """
        Fits the LightGBM model to the training data.

        Parameters:
        - train_data (pd.DataFrame): Training data.

        Returns:
        - None
        """
        split_parameters = self.config['modeling']['split_parameters']
        X, y = self._prepare_data(train_data, **split_parameters)

        model_parameters = self.config['modeling']['model_parameters']
        self.model = lgb.LGBMRegressor(**model_parameters).fit(X, y)

    @Service.log_output
    def predict_outputs(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the target variable on the test data.

        Parameters:
        - test_data (pd.DataFrame): Test data.

        Returns:
        - pd.DataFrame: Test data with predicted target variable.
        """
        split_parameters = self.config['modeling']['split_parameters']
        X_test, _ = self._prepare_data(test_data, **split_parameters)
        y_pred = self.model.predict(X_test)

        test_data[split_parameters['target_variable']] = y_pred
        return test_data

    def calculate_regression_metrics(self, model_name: str, y_test, y_pred, display=True) -> Tuple[float, float, float, float, float, float]:
        """
        Calculates and displays regression metrics for model evaluation.

        Parameters:
        - model_name (str): Name of the model.
        - y_test: True values of the target variable.
        - y_pred: Predicted values of the target variable.
        - display (bool): Whether to print the results.

        Returns:
        - Tuple[float, float, float, float, float, float]: Regression metrics (MAE, MSE, RMSE, R2, MAPE, WMAPE).
        """
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        wmape = mean_absolute_percentage_error(y_test, y_pred, sample_weight=y_test)

        if display:
            print(f"{model_name} model results:")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Mean Squared Error (MSE): {mse:.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"R-squared (R2): {r2:.2f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
            print(f"Weighted Mean Absolute Percentage Error (WMAPE): {wmape * 100:.2f}%")

        return mae, mse, rmse, r2, mape, wmape