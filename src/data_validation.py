from src.service import Service
from sklearn.impute import KNNImputer
import pandas as pd
from typing import Dict, Union, List, Tuple


class DataValidation:
    def __init__(self, service_instance: Service):
        """
        Initializes the DataValidation instance.

        Parameters:
        - service_instance (Service): An instance of the Service class.
        """
        self.service = service_instance
        self.config = self.service.get_config()
        self.imputer = None  # Initialized to None for lazy loading of KNNImputer


    @Service.log_output
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the data validation process.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Processed train_data and test_data.
        """
        # get data paths, read data
        data_paths = self.config['data_validation']['read_path']
        train_data = self.service.read_csv(data_paths['train_data'])
        test_data = self.service.read_csv(data_paths['test_data'])

        # get mapping parameters, map column names
        column_mapping = self.config['data_validation']['column_mapping']
        train_data = self._map_column_names(train_data, column_mapping)
        test_data = self._map_column_names(test_data, column_mapping)

        # get replacement parameters, run replacement
        replacement_dict = self.config['data_validation']['replacement_dict']
        train_data = self._replace_column_names(train_data, replacement_dict)
        test_data = self._replace_column_names(test_data, replacement_dict)

        # get categorical columns and validate dtypes
        categorical_columns = self.config['data_validation']['categorical_columns']
        train_data = self._validate_dtypes(train_data, categorical_columns)
        test_data = self._validate_dtypes(test_data, categorical_columns)

        # remove duplicates
        train_data = self._remove_duplicates(train_data)
        test_data = self._remove_duplicates(test_data)

        # get imputation columns and run imputation
        imputation_parameters = self.config['data_validation']['imputation']
        train_data = self._impute_missing_values(train_data, **imputation_parameters)
        test_data = self._impute_missing_values(test_data, **imputation_parameters)

        # get filtering parameters and run filtering
        columns_to_include = self.config['data_validation']['columns_to_include']
        train_data = self._filter_columns(train_data, columns_to_include)
        test_data = self._filter_columns(test_data, columns_to_include)

        return train_data, test_data

    def _map_column_names(self, dataframe: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Maps column names in the DataFrame according to the specified column_mapping.

        Parameters:
        - dataframe (pd.DataFrame): Input DataFrame.
        - column_mapping (Dict[str, str]): Mapping of old column names to new column names.

        Returns:
        - pd.DataFrame: DataFrame with updated column names.
        """
        return dataframe.rename(columns=column_mapping)

    def _replace_column_names(self, dataframe: pd.DataFrame, replacement_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Replaces specified patterns in column names of the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): Input DataFrame.
        - replacement_dict (Dict[str, str]): Dictionary specifying patterns to replace and their replacements.

        Returns:
        - pd.DataFrame: DataFrame with updated column names.
        """
        for to_replace, replacement in replacement_dict.items():
            dataframe.columns = dataframe.columns.str.replace(to_replace, replacement).str.lower()
        return dataframe

    def _validate_dtypes(self, dataframe: pd.DataFrame, categorical_columns: Union[str, List[str]]) -> pd.DataFrame:
        """
        Validates and converts specified columns to categorical dtype in the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): Input DataFrame.
        - categorical_columns (Union[str, List[str]]): Column(s) to convert to categorical dtype.

        Returns:
        - pd.DataFrame: DataFrame with specified columns converted to categorical dtype.
        """
        dataframe[categorical_columns] = dataframe[categorical_columns].astype('category')
        return dataframe

    def _remove_duplicates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with duplicate rows removed.
        """
        return dataframe.drop_duplicates()

    def _impute_missing_values(self, dataframe: pd.DataFrame, columns_to_impute: List[str], n_neighbors: int,
                               method: str = 'fit_transform') -> pd.DataFrame:
        """
        Imputes missing values in specified columns using KNNImputer.

        Parameters:
        - dataframe (pd.DataFrame): Input DataFrame.
        - columns_to_impute (List[str]): Columns with missing values to impute.
        - n_neighbors (int): Number of neighbors to consider in KNNImputer.
        - method (str): Imputation method, 'fit_transform' for training data, 'transform' for test data.

        Returns:
        - pd.DataFrame: DataFrame with missing values imputed.
        """
        numeric_df = dataframe.select_dtypes(include='number')

        if not self.imputer:
            self.imputer = KNNImputer(n_neighbors=n_neighbors)

        if method == 'fit_transform':
            dataframe[columns_to_impute] = self.imputer.fit_transform(numeric_df[columns_to_impute])
        if method == 'transform':
            dataframe[columns_to_impute] = self.imputer.transform(numeric_df[columns_to_impute])

        return dataframe

    def _filter_columns(self, dataframe: pd.DataFrame, columns_to_include: List[str]) -> pd.DataFrame:
        """
        Filters the DataFrame to include only specified columns.

        Parameters:
        - dataframe (pd.DataFrame): Input DataFrame.
        - columns_to_include (List[str]): Columns to include in the filtered DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with only specified columns.
        """
        return dataframe[columns_to_include]
