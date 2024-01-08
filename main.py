from src.service import Service
from src.data_validation import DataValidation
from src.modeling import Modeling
from src.output_analysis import OutputAnalysis

import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    # Initialize Service
    service_instance = Service()

    # Data Validation
    data_validation = DataValidation(service_instance)
    train_data, test_data = data_validation.run()

    # Model Training and Prediction
    modeling = Modeling(service_instance)
    modeling.fit_model(train_data)
    test_data = modeling.predict_outputs(test_data)

    # Output Analysis
    output_analysis = OutputAnalysis(service_instance)
    
    # Calculate and print weighted cluster average
    weighted_cluster_avg = output_analysis.calculate_weighted_cluster_average(test_data, save_results=True)
    print(weighted_cluster_avg)

if __name__ == "__main__":
    # Execute the main function
    main()
