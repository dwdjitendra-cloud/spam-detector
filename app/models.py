import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Tuple, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Enhanced text preprocessing with financial spam patterns
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text string
        
    Example:
        >>> preprocess_text("Hello! Please verify your account: http://example.com")
        'hello verifi account'
    """
    if pd.isna(text):
        return ""
    
    try:
        text = str(text).lower()
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Keep basic punctuation
        text = re.sub(r'[^\w\s.!?]', '', text)
        # Custom stopwords
        financial_stopwords = ['account', 'bank', 'kyc', 'verify', 'verification', 'click']
        stop = set(stopwords.words('english') + ['ur', 'u', 'im', 'da'] + financial_stopwords)
        text = " ".join(word for word in text.split() if word not in stop)
        # Stemming
        stemmer = PorterStemmer()
        text = " ".join(stemmer.stem(word) for word in text.split())
        return text
    except Exception as e:
        logger.error(f"Text preprocessing failed: {str(e)}")
        return ""

def train_model(data_path: str) -> Pipeline:
    """
    Train and return a spam detection model with robust NaN handling
    """
    try:
        # Load data with explicit NA handling
        df = pd.read_csv(data_path, dtype={'text': 'string', 'label': 'string'})
        
        # Check for missing values
        if df.isna().sum().sum() > 0:
            logger.warning(f"Found missing values:\n{df.isna().sum()}")
            
            # Handle missing text
            df['text'] = df['text'].fillna('')
            
            # Handle missing labels by dropping those rows
            initial_count = len(df)
            df = df.dropna(subset=['label'])
            logger.info(f"Dropped {initial_count - len(df)} rows with missing labels")
            
            if len(df) == 0:
                raise ValueError("No valid data remaining after NaN removal")

        # Feature engineering
        url_pattern = r'http[s]?://\S+|www\.\S+|bit\.ly/\S+'
        df['has_url'] = df['text'].str.contains(url_pattern, case=False).fillna(False).astype(int)
        
        urgency_pattern = r'urgent|immediately|today only|expire|action required'
        df['has_urgent'] = df['text'].str.contains(urgency_pattern, case=False).fillna(False).astype(int)
        
        financial_pattern = r'bank|kyc|verify|account|password'
        df['has_financial'] = df['text'].str.contains(financial_pattern, case=False).fillna(False).astype(int)
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(preprocess_text)
        df['combined_features'] = (
            df['processed_text'] + ' ' +
            df['has_url'].astype(str) + ' ' +
            df['has_urgent'].astype(str) + ' ' +
            df['has_financial'].astype(str)
        )

        # Verify no NaN in features or labels
        if df['combined_features'].isna().any() or df['label'].isna().any():
            raise ValueError("NaN values detected in features or labels after preprocessing")

        # Build and train model
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
            ('clf', MultinomialNB(alpha=0.05))
        ])
        
        model.fit(df['combined_features'], df['label'])
        logger.info(f"Model trained on {len(df)} samples")
        return model
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def save_model(model: Pipeline, model_path: str) -> None:
    """Save trained model to disk with metadata"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': model,
            'version': '1.2',
            'features': ['url', 'urgent', 'financial'],
            'timestamp': pd.Timestamp.now()
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

def load_model(model_path: str) -> Pipeline:
    """Load trained model from disk"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model_data = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model_data['model']
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
def predict_spam(model, text):
    """Make prediction with enhanced features"""
    if not text or pd.isna(text):
        return "ham", 0.0, {'has_url': False, 'has_urgent': False, 'has_financial': False}
        
    try:
        # Convert to string if not already
        text = str(text)
        
        # Feature detection (case insensitive)
        has_url = bool(re.search(
            r'http[s]?://|www\.|bit\.ly|/\S+',
            text.lower()
        ))
        
        has_urgent = bool(re.search(
            r'\b(urgent|immediately|asap|right away|hurry|limited time)\b',
            text.lower()
        ))
        
        has_financial = bool(re.search(
            r'\b(bank|kyc|verify|account|password|card|security|login)\b', 
            text.lower()
        ))
        
        # Combine features for prediction
        processed_text = preprocess_text(text)
        combined_features = f"{processed_text} {int(has_url)} {int(has_urgent)} {int(has_financial)}"
        
        prediction = model.predict([combined_features])[0]
        probability = float(model.predict_proba([combined_features])[0][1])
        
        return prediction, probability, {
            'has_url': has_url,
            'has_urgent': has_urgent,
            'has_financial': has_financial
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "error", 0.0, {'has_url': False, 'has_urgent': False, 'has_financial': False}

def evaluate_model(data_path: str) -> None:
    """Evaluate model performance on test set"""
    try:
        df = pd.read_csv(data_path, dtype={'text': 'string', 'label': 'category'})
        df = add_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['combined_features'], df['label'], test_size=0.2, random_state=42
        )
        
        model = train_model(data_path)
        predictions = model.predict(X_test)
        
        print(classification_report(y_test, predictions))
        return model
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to dataframe"""
    try:
        df['text'] = df['text'].fillna('')
        
        url_pattern = r'http[s]?://\S+|www\.\S+|bit\.ly/\S+'
        urgency_pattern = r'urgent|immediately|today only|expire|action required'
        financial_pattern = r'bank|kyc|verify|account|password'
        
        df['has_url'] = df['text'].str.contains(url_pattern, case=False).astype(int)
        df['has_urgent'] = df['text'].str.contains(urgency_pattern, case=False).astype(int)
        df['has_financial'] = df['text'].str.contains(financial_pattern, case=False).astype(int)
        
        df['processed_text'] = df['text'].apply(preprocess_text)
        df['combined_features'] = (
            df['processed_text'] + ' ' +
            df['has_url'].astype(str) + ' ' +
            df['has_urgent'].astype(str) + ' ' +
            df['has_financial'].astype(str)
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise