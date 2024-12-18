'''
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    input_data_path: str = os.path.join('artifacts/Feature_engineering', "processed_data.csv")
    validation_report_path: str = os.path.join('artifacts/Data_validation', "validation_report.txt")
    validation_data_path: str = os.path.join('artifacts/Data_validation', "validation_data.csv")

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def load_data(self):
        """Load the dataset for validation."""
        logging.info("Loading data for validation")
        try:
            df = pd.read_csv(self.validation_config.input_data_path)
            logging.info("Data loaded successfully")
            return df
        except Exception as e:
            logging.error("Error loading data: {}".format(e))
            raise CustomException(e, sys)

    def validate_data(self, df):
        """Perform various validation checks on the dataset."""
        validations = {}
        
        # Check for duplicates
        validations['No Duplicates'] = df.duplicated().sum() == 0

            # Initialize a dictionary to hold missing value counts
        missing_values_count = df.isnull().sum()

        # Check and handle missing values for all columns
        for column in df.columns:
            if missing_values_count[column] > 0:
                if missing_values_count[column] < 25:
                    logging.info(f"Dropping column '{column}' with {missing_values_count[column]} missing values.")
                    df.drop(column, axis=1, inplace=True)
                else:
                    logging.info(f"Imputing column '{column}' with {missing_values_count[column]} missing values using median.")
                    median_value = df[column].median()
                    df[column].fillna(median_value, inplace=True)

        # Update validation results after handling missing values
        validations['No Missing Values'] = df.isnull().sum().sum() == 0
    
        # Check non-negative energy supply and log the count of non-negative values
        non_negative_supply_count = (df["Energy Supplied to Grid (kWh)"] >= 0).sum()
        total_supply_count = df.shape[0]
        validations['Non-negative Energy Supplied'] = non_negative_supply_count == total_supply_count
        logging.info(f"Number of non-negative energy supplied: {non_negative_supply_count} out of {total_supply_count}")
        
        
        validations['Energy Supplied <= Total Generated'] = (
            df["Energy Supplied to Grid (kWh)"] <= 
            (df["Solar Energy Generated (kWh)"] + df["Wind Energy Generated (kWh)"])
        ).all()

        # Check energy conservation and log results
        energy_generated = df["Solar Energy Generated (kWh)"] + df["Wind Energy Generated (kWh)"]
        energy_stored_and_supplied = df["Energy Supplied to Grid (kWh)"] + df["Energy Wasted (kWh)"]

        # Log values for debugging
        for i in range(len(df)):
            if not np.isclose(energy_generated[i], energy_stored_and_supplied[i], atol=1e-2):
                logging.warning(f"Energy Conservation Check Failed at index {i}: "
                                f"Generated: {energy_generated[i]}, "
                                f"Stored/Supplied/Wasted: {energy_stored_and_supplied[i]}")

        validations['Energy Conservation'] = np.all(np.isclose(energy_generated, energy_stored_and_supplied, atol=1e-2))

        return validations
    
    
        # # Check energy conservation
        # validations['Energy Conservation'] = (
        #     (df["Solar Energy Generated (kWh)"] + df["Wind Energy Generated (kWh)"]) 
        #     == (df["Current Storage Level (kWh)"] + df["Energy Supplied to Grid (kWh)"] + df["Energy Wasted (kWh)"])
        # ).all()
        
        # return validations

    def save_validation_report(self, validations, df):
        """Save the validation report."""
        try:
            with open(self.validation_config.validation_report_path, 'w') as f:
                for check, result in validations.items():
                    f.write(f"{check}: {'Passed' if result else 'Failed'}\n")
                    
            logging.info(f"Validation report saved to {self.validation_config.validation_report_path}")
            
            # Save the validated data to a CSV file
            df.to_csv(self.validation_config.validation_data_path, index=False)
            logging.info(f"Validated data saved to {self.validation_config.validation_data_path}")
        except Exception as e:
            logging.error("Error saving validation report: {}".format(e))
            raise CustomException(e, sys)

    def run_validation_pipeline(self):
        """Run the complete data validation pipeline."""
        df = self.load_data()
        validations = self.validate_data(df)
        self.save_validation_report(validations, df)

if __name__ == "__main__":
    obj = DataValidation()
    obj.run_validation_pipeline()
'''

# ----------------------------------------------


import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    input_data_path: str = os.path.join('artifacts/Feature_engineering', "processed_data.csv")
    validation_report_path: str = os.path.join('artifacts/Data_validation', "validation_report.txt")
    validation_data_path: str = os.path.join('artifacts/Data_validation', "validation_data.csv")

class DataValidation:
    def __init__(self):
        self.config = DataValidationConfig()
        self.df = self.load_data()
        os.makedirs(os.path.dirname(self.config.validation_report_path), exist_ok=True)  
        os.makedirs(os.path.dirname(self.config.validation_data_path), exist_ok=True)  

    def load_data(self):
        """Load the dataset from the specified path."""
        logging.info(f"Loading data from {self.config.input_data_path}")
        try:
            df = pd.read_csv(self.config.input_data_path)
            logging.info("Data loaded successfully")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomException(e, sys)

    def check_missing_values(self):
        """Check for missing values in the DataFrame."""
        missing_values = self.df.isnull().sum()
        missing_columns = missing_values[missing_values > 0]

        if not missing_columns.empty:
            logging.warning("Missing values detected in the following columns:")
            for column, count in missing_columns.items():
                logging.warning(f"{column}: {count} missing values")
            return True  # Missing values found
        else:
            logging.info("No missing values detected.")
            return False  # No missing values

    def validate_storage_level(self):
        """Check if the current storage level is within capacity."""
        is_within_capacity = (self.df["Current Storage Level (kWh)"] <= self.df["Storage Capacity (kWh)"]).all()
        logging.info(f"Storage level within capacity: {is_within_capacity}")
        return is_within_capacity

    def validate_non_negative_energy_waste(self):
        """Check if energy waste is non-negative."""
        non_negative_waste = (self.df["Energy Wasted (kWh)"] >= 0).all()
        logging.info(f"Non-negative energy waste: {non_negative_waste}")
        return non_negative_waste

    def validate_non_negative_grid_supply(self):
        """Check if energy supplied to the grid is non-negative."""
        non_negative_supply = (self.df["Energy Supplied to Grid (kWh)"] >= 0).all()
        logging.info(f"Non-negative grid supply: {non_negative_supply}")
        return non_negative_supply

    def generate_validation_report(self):
        """Generate a validation report."""
        with open(self.config.validation_report_path, 'w') as report_file:
            report_file.write("Validation Report:\n")
            
            # Check for missing values
            if self.check_missing_values():
                report_file.write("Missing Values Detected:\n")
                report_file.write(str(self.df.isnull().sum()))
            
            # Validate other conditions
            report_file.write("\nStorage Level Validation:\n")
            report_file.write(f"Within Capacity: {self.validate_storage_level()}\n")

            report_file.write("\nNon-Negative Energy Waste Validation:\n")
            report_file.write(f"Non-Negative Waste: {self.validate_non_negative_energy_waste()}\n")

            report_file.write("\nNon-Negative Grid Supply Validation:\n")
            report_file.write(f"Non-Negative Supply: {self.validate_non_negative_grid_supply()}\n")

    def save_validation_data(self):
        """Save the validation data to a CSV file."""
        self.df.to_csv(self.config.validation_data_path, index=False)
        logging.info(f"Validation data saved to {self.config.validation_data_path}")

if __name__ == "__main__":
    validator = DataValidation()
    validator.generate_validation_report()
    validator.save_validation_data()
