import pandas as pd
import yaml
from functools import wraps
import time

class Service:
    """
    A service class for reading and writing CSV files and handling configuration using pandas and YAML.

    Parameters:
    - config_file_path (str): Path to the YAML configuration file.

    Attributes:
    - config (dict): Configuration settings loaded from the YAML file.
    """

    def __init__(self, config_file_path:str='config.yaml'):
        """
        Initializes the FileService instance.

        Parameters:
        - config_file_path (str): Path to the YAML configuration file.
        """
        self.config = self._read_config(config_file_path)

    @staticmethod
    def log_output(func):
        """
        A decorator to log the execution time of a method.

        Parameters:
        - func (Callable): The function to be decorated.

        Returns:
        - Callable: The decorated function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            A decorator to log the execution time of a method.

            Parameters
            ----------
            func : function
                The function to be decorated.

            Returns
            -------
            function
                The decorated function.
            """
            class_name = args[0].__class__.__name__
            func_name = func.__name__

            start_time = time.time()
            print(f"{class_name}.{func_name} started")
            result = func(*args, **kwargs)
            end_time = time.time()
  
            execution_time = end_time - start_time
            print(f"{class_name}.{func_name} finished. Took {execution_time:.2f} seconds")
            return result
        return wrapper

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file using pandas.

        Parameters:
        - file_path (str): Path to the CSV file.

        Returns:
        pd.DataFrame: DataFrame containing the CSV data.
        """
        data = pd.read_csv(file_path)
        return data

    def write_csv(self, file_path: str, data: pd.DataFrame) -> None:
        """
        Writes a DataFrame to a CSV file using pandas.

        Parameters:
        - file_path (str): Path to the output CSV file.
        - data (pd.DataFrame): DataFrame to be written to the CSV file.

        Returns:
        None
        """
        data.to_csv(file_path, index=False)

    def _read_config(self, config_file_path: str) -> dict:
        """
        Reads the YAML configuration file.

        Parameters:
        - config_file_path (str): Path to the YAML configuration file.

        Returns:
        dict: Configuration settings loaded from the YAML file.
        """
        with open(config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config
    
    def get_config(self) -> dict:
        """
        Retrieves the current configuration settings.

        Returns:
        dict: Configuration settings.
        """
        return self.config