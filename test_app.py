#!/usr/bin/env python3
"""
Test script for MindCare Mental Health Assessment App
"""

import requests
import json
import time

def test_application():
    """Test the complete application"""
    base_url = "http://localhost:5000"
    
    print("Testing MindCare Application...")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"Health check passed: {health_data}")
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False
    
    # Test 2: Model info
    print("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print(f"Model info: {model_info}")
        else:
            print(f"Model info failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Model info failed: {e}")
    
    # Test 3: Prediction
    print("\n3. Testing prediction endpoint...")
    test_data = {
        "gender": "Female",
        "country": "United States",
        "occupation": "Corporate",
        "self_employed": "No",
        "family_history": "No",
        "treatment": "No",
        "days_indoors": "1-14 days",
        "growing_stress": "No",
        "changes_habits": "No",
        "mental_health_history": "No",
        "mood_swings": "Low",
        "coping_struggles": "No",
        "work_interest": "Yes",
        "social_weakness": "No",
        "mental_health_interview": "Yes"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict", 
            json=test_data, 
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        if response.status_code == 200:
            prediction = response.json()
            print(f"Prediction successful:")
            print(f"   Prediction: {prediction['prediction']}")
            print(f"   Confidence: {prediction['confidence']:.2%}")
            print(f"   Recommendations: {len(prediction['recommendations'])} items")
        else:
            print(f"Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Prediction failed: {e}")
    
    print("\n" + "=" * 50)
    print("Testing completed!")
    return True

if __name__ == "__main__":
    # Wait a moment for the server to start
    print("Waiting for server to start...")
    time.sleep(3)
    test_application()
