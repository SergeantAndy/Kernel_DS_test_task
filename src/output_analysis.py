from src.service import Service
import pandas as pd
from typing import Dict


class OutputAnalysis:
    def __init__(self, service_instance: Service):
        """
        Initializes the OutputAnalysis instance.

        Parameters:
        - service_instance (Service): An instance of the Service class.
        """
        self.service = service_instance
        self.config = self.service.get_config()

    def calculate_weighted_cluster_average(self, data: pd.DataFrame, save_results: bool = True) -> pd.Series:
        """
        Calculate the weighted cluster average based on area, yield, and field id.

        Parameters:
        - data (pd.DataFrame): Input DataFrame containing columns 'cluster', 'area', 'yield', and 'field'.
        - save_results (bool): Flag indicating whether to save the results. Default is True.

        Returns:
        - pd.Series: Weighted cluster averages for each cluster.
        """
        # Check if the required columns are present in the input data
        required_columns = self.config['output_analysis']['wca_parameters']

        # Calculate weighted cluster averages
        weighted_avg = (
            data.groupby('cluster')
                .apply(lambda cluster_data: 
                    (cluster_data[required_columns['area']] * cluster_data[required_columns['yield']]).sum() / cluster_data[required_columns['area']].sum())
                .rename('wca')
        )

        result = pd.merge(data, weighted_avg, left_on='cluster', right_index=True)

        if save_results:
            output_path = self.config['output_analysis']['output_path']
            self.service.write_csv(output_path, result)

        return result
