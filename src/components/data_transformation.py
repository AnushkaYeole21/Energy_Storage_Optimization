import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class FeatureEngineeringConfig:
    os.makedirs('artifacts/Feature_engineering', exist_ok=True)
    input_data_path: str = os.path.join('artifacts/Data_ingestion', "data.csv")
    output_data_path: str = os.path.join('artifacts/Feature_engineering', "processed_data.csv")
    scaler_path: str = os.path.join('artifacts/Feature_engineering', "scaler.pkl")
    all_features_path: str = os.path.join('artifacts/Feature_engineering', "all_features.csv")

class FeatureEngineering:
    def __init__(self):
        self.feature_engineering_config = FeatureEngineeringConfig()
        os.makedirs(os.path.dirname(self.feature_engineering_config.output_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.feature_engineering_config.all_features_path), exist_ok=True)

    def load_data(self):
        """Load the dataset from the specified path."""
        logging.info("Loading data from {}".format(self.feature_engineering_config.input_data_path))
        try:
            df = pd.read_csv(self.feature_engineering_config.input_data_path)
            logging.info("Data loaded successfully")
            return df
        except Exception as e:
            logging.error("Error loading data: {}".format(e))
            raise CustomException(e, sys)

    def handle_missing_values(self, df):
        """Handle missing values in the DataFrame."""
        try:
            # Log the number of missing values for each column before handling them
            missing_values_count = df.isnull().sum()
            logging.info("Missing values detected in the dataset:")
            for column, count in missing_values_count.items():
                logging.info(f"{column}: {count} missing values")

            # Fill missing values with median for numerical columns
            for column in df.select_dtypes(include=[np.number]).columns:
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                logging.info(f"Missing values in '{column}' filled with median value: {median_value}")
            
            return df
        except Exception as e:
            logging.error("Error handling missing values: {}".format(e))
            raise CustomException(e, sys)

    def extract_time_features(self, df):
        """Extract time-based features from the timestamp."""
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Hour'] = df['Timestamp'].dt.hour
            df['Weekday'] = df['Timestamp'].dt.weekday
            df['Month'] = df['Timestamp'].dt.month
            logging.info("Time-based features created successfully")
            return df
        except Exception as e:
            logging.error("Error extracting time features: {}".format(e))
            raise CustomException(e, sys)

    def create_lag_features(self, df):
        """Create lag features for wind and solar energy generated."""
        try:
            df['Wind_Lag_1'] = df['Wind Energy Generated (kWh)'].shift(1)
            df['Solar_Lag_1'] = df['Solar Energy Generated (kWh)'].shift(1)
            logging.info("Wind_Lag_1 and Solar_Lag_1 features created successfully")    
            df.dropna(subset=['Wind_Lag_1', 'Solar_Lag_1'], inplace=True)
            logging.info("Rows with null values in lag features dropped.")
            return df
        except Exception as e:
            logging.error("Error creating lag features: {}".format(e))
            raise CustomException(e, sys)

    def calculate_metrics(self, df):
        """Calculate additional metrics such as total demand and energy efficiency."""
        try:
            df['Total_Demand'] = (df['Residential Demand (kWh)'] + 
                                  df['Commercial Demand (kWh)'] + 
                                  df['Industrial Demand (kWh)'])
                                  
            # Avoid division by zero by replacing 0 with NaN before division
            df['Energy_Efficiency'] = (df['Energy Supplied to Grid (kWh)'] /
                                        (df['Total_Demand'].replace(0, np.nan)))
                                        
            logging.info("Energy_Efficiency calculated successfully")
            return df
        except Exception as e:
            logging.error("Error calculating metrics: {}".format(e))
            raise CustomException(e, sys)

    def calculate_additional_metrics(self, df):
        """Calculate additional metrics such as total energy generated and energy balance."""
        try:
            # Step 1: Calculate total energy generated
            df["Total Energy Generated (kWh)"] = (
                df["Solar Energy Generated (kWh)"] + 
                df["Wind Energy Generated (kWh)"]
            )

            # Step 2: Calculate energy stored (change in storage level)
            df["Energy Stored (kWh)"] = (
                df["Current Storage Level (kWh)"] - 
                df["Current Storage Level (kWh)"].shift(1, fill_value=0)
            )

            # Step 3: Verify energy conservation
            df["Energy Balance"] = (
                df["Total Energy Generated (kWh)"] -
                df["Energy Stored (kWh)"] -
                df["Energy Supplied to Grid (kWh)"] -
                df["Energy Wasted (kWh)"]
            )

            # Step 4: Check if energy balance is approximately zero
            tolerance = 1e-3  # Small value for floating-point tolerance
            energy_conservation_violations = np.abs(df["Energy Balance"]) > tolerance

            if energy_conservation_violations.any():
                logging.warning("Energy conservation violated in some rows.")
                logging.info(df.loc[energy_conservation_violations, ["Total Energy Generated (kWh)", "Energy Stored (kWh)", "Energy Balance"]])
            else:
                logging.info("Energy conservation holds for all rows.")

            logging.info("Additional metrics calculated successfully")
            return df
        except Exception as e:
            logging.error("Error calculating additional metrics: {}".format(e))
            raise CustomException(e, sys)

    def save_processed_data(self, df):
        """Save the processed DataFrame to a CSV file."""
        logging.info(f"Saving processed data to {self.feature_engineering_config.output_data_path}")
        try:
            os.makedirs(os.path.dirname(self.feature_engineering_config.output_data_path), exist_ok=True)
            df.to_csv(self.feature_engineering_config.output_data_path, index=False)
            logging.info("Processed data saved successfully")
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")
            raise CustomException(e, sys)

    def save_all_features(self, df):
        """Save all features DataFrame to a separate CSV file."""
        logging.info(f"Saving all features to {self.feature_engineering_config.all_features_path}")
        try:
            os.makedirs(os.path.dirname(self.feature_engineering_config.all_features_path), exist_ok=True)
            df.to_csv(self.feature_engineering_config.all_features_path, index=False)
            logging.info("All features saved successfully")
        except Exception as e:
            logging.error(f"Error saving all features: {e}")
            raise CustomException(e, sys)

    def standardize_features(self, df):
        """Standardize the numerical features."""
        try:
           numerical_cols = [
                "Temperature (°C)", "Wind Speed (m/s)", "Solar Radiation (W/m²)", "Cloud Cover (%)", "Precipitation (mm)",
                "Residential Demand (kWh)", "Commercial Demand (kWh)", "Industrial Demand (kWh)", "Hour", "Weekday", "Month",
                "Wind_Lag_1", "Solar_Lag_1", "Total Energy Generated (kWh)", "Energy Stored (kWh)", 
                "Total_Demand", "Energy_Efficiency"
           ]

           scaler = StandardScaler()
           
           # Standardize the numerical columns
           scaled_values = scaler.fit_transform(df[numerical_cols])
           
           # Update DataFrame with standardized values
           df[numerical_cols] = scaled_values
            
           # Save the fitted scaler as a .pkl file
           joblib.dump(scaler, self.feature_engineering_config.scaler_path)
           
           logging.info(f"Scaler saved to {self.feature_engineering_config.scaler_path}")

           return df

        except Exception as e:
           logging.error("Error during feature standardization: {}".format(e))
           raise CustomException(e, sys)


    def feature_engineering_pipeline(self):
        """Run the complete feature engineering pipeline."""
        # Load data
        df = self.load_data()
        
        # Handle missing values first
        df = self.handle_missing_values(df)

        # Feature engineering steps
        df = self.extract_time_features(df)
        df = self.create_lag_features(df)
        df = self.calculate_metrics(df)
        df = self.calculate_additional_metrics(df)

        # Standardize features and save processed data
        standardized_df = self.standardize_features(df)

        # Save processed data
        self.save_processed_data(df)
        self.save_all_features(df)

        # Save standardized data to processed_data.csv after standardization is done.
        self.save_processed_data(standardized_df)

if __name__ == "__main__":
    obj = FeatureEngineering()
    obj.feature_engineering_pipeline()
