import pandas as pd
from app.models import train_model, save_model, evaluate_model
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(input_path: str, output_path: str) -> bool:
    """Clean and preprocess the raw data"""
    try:
        logger.info(f"Cleaning data from {input_path}")
        
        df = pd.read_csv(input_path, dtype={'text': 'string', 'label': 'string'})
        initial_rows = len(df)
        
        # Data cleaning steps
        df = df.dropna(subset=['label'])  # Remove rows with missing labels
        df['text'] = df['text'].fillna('')  # Fill missing text with empty strings
        df = df.drop_duplicates()  # Remove duplicates
        
        # Validate labels
        valid_labels = ['ham', 'spam']
        df = df[df['label'].isin(valid_labels)]
        
        # Save cleaned data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Data cleaning complete. Removed {initial_rows - len(df)} rows.")
        return True
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {str(e)}")
        return False

def main():
    # File paths
    raw_data_path = 'data/spam_ham_dataset.csv'
    clean_data_path = 'data/cleaned_spam_ham.csv'
    model_path = 'app/spam_model.joblib'
    
    try:
        # Step 1: Clean the data
        if not clean_data(raw_data_path, clean_data_path):
            raise RuntimeError("Data cleaning failed")
        
        # Step 2: Train the model
        logger.info("Training model...")
        model = train_model(clean_data_path)
        
        # Step 3: Save the model
        logger.info("Saving model...")
        save_model(model, model_path)
        
        # Step 4: Evaluate
        logger.info("Evaluating model...")
        evaluate_model(clean_data_path)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()