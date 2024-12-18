import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb
from src.exception import CustomException
from src.logger import logging

class ModelTraining:
    def __init__(self):
        self.data_path = os.path.join('artifacts/Data_validation', "validation_data.csv")  
        self.model_path = os.path.join('artifacts/Model_training', "energy_model.pkl")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)  

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
        # Define the target column
        target_column = 'Energy Supplied to Grid (kWh)'

        # Separate features (X) and target (y)
        X = df.drop(columns=[target_column, 'Energy Wasted (kWh)', 'Timestamp'])  
        y = df[target_column]

        # Split the data into training and testing (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        logging.info("Data preprocessed and split into training and testing sets")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Train the machine learning model."""
        # Initialize the XGBoost Regressor
        model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3,
                                  learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)

        # Fit the model to the training data
        model.fit(X_train, y_train)

        logging.info("Model trained successfully")
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the trained model."""
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        return rmse

    def save_model(self, model):
        """Save the trained model to a file."""
        joblib.dump(model, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        df = self.load_data()
        
        X_train, X_test, y_train, y_test = self.preprocess_data(df)
        
        model = self.train_model(X_train, y_train)
        
        rmse = self.evaluate_model(model, X_test, y_test)
        
        self.save_model(model)

if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.run_training_pipeline()
