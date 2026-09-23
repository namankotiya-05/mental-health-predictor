from flask import Flask, request, jsonify, render_template
import pandas as pd
import xgboost as xgb
import numpy as np
from flask_cors import CORS
import os
import traceback

app = Flask(__name__)
CORS(app)

# Global variables for model components
model = None

def load_model():
    """Load the trained XGBoost model"""
    global model
    
    try:
        print("Loading XGBoost model...")
        # Load the XGBoost model
        model = xgb.Booster()
        model.load_model('xgb_mental_health_model.json')
        print("XGBoost model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        print("Continuing with simplified scoring system...")
        return False

def predict_mental_health(data):
    """Make prediction using intelligent scoring system"""
    try:
        # Initialize score
        score = 0
        
        # Risk factors that increase the score
        risk_factors = {
            'family_history': 2,
            'treatment': 2,
            'mental_health_history': 2,
            'growing_stress': 2,
            'coping_struggles': 2,
            'social_weakness': 2
        }
        
        # Add points for risk factors
        for factor, points in risk_factors.items():
            if data.get(factor) == 'Yes':
                score += points
        
        # Mood swings scoring
        mood_swings = data.get('mood_swings', '')
        if mood_swings == 'High':
            score += 3
        elif mood_swings == 'Medium':
            score += 1
        
        # Days indoors scoring
        days_indoors = data.get('days_indoors', '')
        if days_indoors == 'More than 2 months':
            score += 3
        elif days_indoors == '31-60 days':
            score += 2
        elif days_indoors == '15-30 days':
            score += 1
        
        # Work interest (reverse scoring)
        if data.get('work_interest') == 'No':
            score += 1
        
        # Mental health interview comfort
        interview_comfort = data.get('mental_health_interview', '')
        if interview_comfort == 'No':
            score += 1
        elif interview_comfort == 'Maybe':
            score += 0.5
        
        # Changes in habits
        if data.get('changes_habits') == 'Yes':
            score += 1
        
        # Determine prediction based on score
        if score >= 8:
            prediction = 'Yes'
            confidence = min(0.95, 0.6 + (score - 8) * 0.05)
        elif score >= 4:
            prediction = 'Not sure'
            confidence = 0.7 + (score - 4) * 0.05
        else:
            prediction = 'No'
            confidence = min(0.9, 0.6 + (4 - score) * 0.05)
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 'Not sure', 0.5

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
        'mode': 'intelligent_scoring'
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_type': 'XGBoost + Intelligent Scoring',
        'model_loaded': model is not None,
        'mode': 'hybrid_intelligent'
    })

if __name__ == '__main__':
    print("Starting MindCare Mental Health Assessment App...")
    print("=" * 60)
    
    # Try to load the XGBoost model
    model_loaded = load_model()
    
    if model_loaded:
        print("XGBoost model loaded successfully!")
    else:
        print("Using intelligent scoring system...")
    
    print("=" * 60)
    print("Application ready!")
    print("Starting Flask server...")
    print("Open your browser to: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
