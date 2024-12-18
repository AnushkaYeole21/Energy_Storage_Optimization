import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model, can be changed to any other model
from sklearn.metrics import mean_squared_error
from src.exception import CustomException
from src.logger import logging

class ModelTraining:
    def __init__(self):
        self.data_path = os.path.join('artifacts/Data_validation', "validation_data.csv")  # Use validation data
        self.model_path = os.path.join('artifacts/Model', "energy_model.pkl")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)  # Ensure the directory exists

    def load_data(self):
        """Load the dataset from the specified path."""
        logging.info(f"Loading data from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            logging.info("Data loaded successfully")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomException(e, sys)

    def preprocess_data(self, df):
        """Preprocess the data for training."""
        # Select features and target variable
        features = df.drop(columns=["Energy Supplied to Grid (kWh)", "Energy Wasted (kWh)", "Timestamp"])  # Adjust based on your target variable
        target = df["Energy Supplied to Grid (kWh)"]  # Example target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        logging.info("Data preprocessed and split into training and testing sets")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Train the machine learning model."""
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # Example model
        model.fit(X_train, y_train)
        logging.info("Model trained successfully")
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the trained model."""
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Model evaluation completed with Mean Squared Error: {mse:.4f}")
        return mse

    def save_model(self, model):
        """Save the trained model to a file."""
        joblib.dump(model, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(df)
        
        model = self.train_model(X_train, y_train)
        
        self.evaluate_model(model, X_test, y_test)
        
        self.save_model(model)

if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.run_training_pipeline()
