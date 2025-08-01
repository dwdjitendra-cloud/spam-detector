from flask import render_template, request, jsonify
from app import app
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'spam_model.joblib')
try:
    model_data = joblib.load(model_path)
    model = model_data['model']
    logger.info("Model loaded successfully")
except Exception as e:
    model = None
    logger.error(f"Failed to load model: {str(e)}")

@app.route('/')
def home():
    return render_template(
        'index.html',
        developer_name="Jitendra Kumar Dodwadiya",
        linkedin_url="https://www.linkedin.com/in/dwdjitendra",  # Changed variable name for consistency
        current_year=datetime.now().year
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text or not isinstance(text, str):
        return jsonify({
            'error': 'Invalid text input',
            'developer': {
                'name': "Jitendra Kumar Dodwadiya",
                'linkedin': "https://www.linkedin.com/in/dwdjitendra"
            }
        }), 400
    
    try:
        text = str(text).lower()
        
        features = {
            'has_url': bool(re.search(
                r'http[s]?://|www\.|bit\.ly|/\S+|https?://\S+',
                text
            )),
            'has_urgent': bool(re.search(
                r'\b(urgent|immediately|asap|right away|hurry|limited time|expiring soon|action required)\b',
                text
            )),
            'has_financial': bool(re.search(
                r'\b(bank|kyc|verify|account|password|card|security|login|credentials|payment|transaction)\b',
                text
            ))
        }
        
        processed_text = preprocess_text(text)
        combined_features = f"{processed_text} {int(features['has_url'])} {int(features['has_urgent'])} {int(features['has_financial'])}"
        
        prediction = model.predict([combined_features])[0]
        probability = float(model.predict_proba([combined_features])[0][1])
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'features': features,
            'developer': {
                'name': "Jitendra Kumar Dodwadiya",
                'linkedin': "https://www.linkedin.com/in/dwdjitendra"
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'developer': {
                'name': "Jitendra Kumar Dodwadiya",
                'linkedin': "https://www.linkedin.com/in/dwdjitendra"
            }
        }), 500

def preprocess_text(text):
    """Enhanced text preprocessing"""
    try:
        nltk.download('stopwords', quiet=True)
        text = str(text).lower()
        text = re.sub(r'\b\d{10,}\b', '', text)
        text = re.sub(r'http[s]?://\S+|www\.\S+|bit\.ly/\S+', '', text)
        text = re.sub(r'[^\w\s.!?]', '', text)
        financial_stopwords = ['account', 'bank', 'kyc', 'verify', 'verification', 'click', 'register']
        stop = set(stopwords.words('english') + ['ur', 'u', 'im', 'da'] + financial_stopwords)
        text = " ".join(word for word in text.split() if word not in stop)
        stemmer = PorterStemmer()
        text = " ".join(stemmer.stem(word) for word in text.split())
        return text
    except Exception as e:
        logger.error(f"Text preprocessing failed: {str(e)}")
        return ""