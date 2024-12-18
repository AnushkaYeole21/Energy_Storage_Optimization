import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logging

class ModelEvaluation:
    def __init__(self):
        self.data_path = os.path.join('artifacts/Data_validation', "validation_data.csv")  # Use validation data
        self.model_path = os.path.join('artifacts/Model', "energy_model.pkl")
        
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

    def load_model(self):
        """Load the trained model from the specified path."""
        logging.info(f"Loading model from {self.model_path}")
        try:
            model = joblib.load(self.model_path)
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException(e, sys)

    def evaluate_model(self, model, df):
        """Evaluate the loaded model using the provided dataset."""
        target_column = 'Energy Supplied to Grid (kWh)'
        
        # Separate features (X) and target (y)
        X = df.drop(columns=[target_column, 'Timestamp', 'Energy Wasted (kWh)'])  # Drop unused columns
        y = df[target_column]

        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y, y_pred)

        # Log metrics
        logging.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logging.info(f"R-squared: {r2:.4f}")

    def run_evaluation_pipeline(self):
        """Run the complete evaluation pipeline."""
        df = self.load_data()
        
        model = self.load_model()
        
        self.evaluate_model(model, df)

if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.run_evaluation_pipeline()
