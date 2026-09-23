from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
from flask_cors import CORS
import os
import traceback
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for model components
model = None
preprocessor = None
label_encoder = None

def create_fallback_preprocessor():
    """Create a fallback preprocessor if the pickle file is incompatible"""
    print("Creating fallback preprocessor...")
    
    # Define categorical and numerical columns based on your dataset
    categorical_columns = [
        'Gender', 'Country', 'Occupation', 'self_employed', 'family_history',
        'treatment', 'Days_Indoors', 'Growing_Stress', 'Changes_Habits',
        'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
        'Work_Interest', 'Social_Weakness', 'mental_health_interview'
    ]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def create_fallback_label_encoder():
    """Create a fallback label encoder"""
    print("Creating fallback label encoder...")
    
    le = LabelEncoder()
    # Fit with the expected classes
    le.fit(['No', 'Yes', 'Not sure'])
    
    return le

def load_model():
    """Load the trained model and preprocessing components"""
    global model, preprocessor, label_encoder
    
    try:
        print("Loading XGBoost model...")
        # Load the XGBoost model
        model = xgb.Booster()
        model.load_model('xgb_mental_health_model.json')
        print("XGBoost model loaded successfully")
        
        print("Loading preprocessor...")
        try:
            # Try to load the preprocessor
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            print("Preprocessor loaded successfully")
        except Exception as e:
            print(f"Preprocessor pickle failed: {e}")
            print("Creating fallback preprocessor...")
            preprocessor = create_fallback_preprocessor()
            # Fit the preprocessor with sample data
            sample_data = pd.read_csv('Mental Health Dataset.csv').head(100)
            preprocessor.fit(sample_data)
            print("Fallback preprocessor created and fitted")
        
        print("Loading label encoder...")
        try:
            # Try to load the label encoder
            with open('label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            print("Label encoder loaded successfully")
        except Exception as e:
            print(f"Label encoder pickle failed: {e}")
            print("Creating fallback label encoder...")
            label_encoder = create_fallback_label_encoder()
            print("Fallback label encoder created")
        
        print("All model components loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def preprocess_input(data):
    """Preprocess the input data for prediction"""
    try:
        # Create a DataFrame with the input data
        df = pd.DataFrame([data])
        
        # Apply preprocessing
        processed_data = preprocessor.transform(df)
        
        return processed_data
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise e

def predict_mental_health(data):
    """Make prediction using the trained model"""
    try:
        # Preprocess the input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction_proba = model.predict(processed_data)
        prediction_class = np.argmax(prediction_proba, axis=1)
        
        # Convert back to original labels
        predicted_label = label_encoder.inverse_transform(prediction_class)[0]
        confidence = float(np.max(prediction_proba))
        
        return predicted_label, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None, 0.0

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for mental health prediction"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'gender', 'country', 'occupation', 'self_employed', 'family_history',
            'treatment', 'days_indoors', 'growing_stress', 'changes_habits',
            'mental_health_history', 'mood_swings', 'coping_struggles',
            'work_interest', 'social_weakness', 'mental_health_interview'
        ]
        
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Make prediction
        prediction, confidence = predict_mental_health(data)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Prepare response
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'recommendations': get_recommendations(prediction, confidence)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def get_recommendations(prediction, confidence):
    """Generate recommendations based on prediction"""
    recommendations = []
    
    if prediction == 'Yes':
        recommendations = [
            "Consider scheduling an appointment with a mental health professional",
            "Contact a mental health hotline if you need immediate support",
            "Reach out to trusted friends or family members",
            "Consider therapy or counseling services",
            "Practice self-care and stress management techniques",
            "Join a support group or community",
            "Consider medication consultation with a psychiatrist"
        ]
    elif prediction == 'Not sure':
        recommendations = [
            "Keep a mood journal to track patterns",
            "Maintain regular exercise and healthy habits",
            "Stay connected with supportive people",
            "Learn about mental health and wellness",
            "Consider a mental health check-up",
            "Practice mindfulness and meditation",
            "Monitor your sleep and eating patterns"
        ]
    else:
        recommendations = [
            "Continue your current self-care practices",
            "Support friends and family who may be struggling",
            "Share mental health awareness with others",
            "Maintain your healthy lifestyle habits",
            "Be a mental health advocate in your community",
            "Consider volunteering for mental health organizations",
            "Stay informed about mental health resources"
        ]
    
    return recommendations

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'label_encoder_loaded': label_encoder is not None
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'XGBoost',
        'preprocessor_loaded': preprocessor is not None,
        'label_encoder_loaded': label_encoder is not None,
        'classes': label_encoder.classes_.tolist() if label_encoder else None
    })

if __name__ == '__main__':
    print("Starting MindCare Mental Health Assessment App...")
    print("=" * 50)
    
    # Load model on startup
    if load_model():
        print("=" * 50)
        print("Model loaded successfully!")
        print("Starting Flask server...")
        print("Open your browser to: http://localhost:5000")
        print("=" * 50)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("=" * 50)
        print("Failed to load model. Please check your model files:")
        print("   - xgb_mental_health_model.json")
        print("   - preprocessor.pkl")
        print("   - label_encoder.pkl")
        print("=" * 50)
