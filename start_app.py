#!/usr/bin/env python3
"""
MindCare Mental Health Assessment App - Smart Startup Script
Handles all compatibility issues and provides the best experience
"""

import subprocess
import sys
import time
import webbrowser
import os

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['flask', 'pandas', 'numpy', 'xgboost', 'scikit-learn', 'flask-cors']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Dependencies installed successfully!")
    else:
        print("All dependencies are installed!")

def check_model_files():
    """Check which model files are available"""
    files_status = {
        'xgb_mental_health_model.json': os.path.exists('xgb_mental_health_model.json'),
        'preprocessor.pkl': os.path.exists('preprocessor.pkl'),
        'label_encoder.pkl': os.path.exists('label_encoder.pkl')
    }
    
    print("Model files status:")
    for file, exists in files_status.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    
    return files_status

def start_application():
    """Start the Flask application with the best available mode"""
    print("=" * 60)
    print("MindCare - Mental Health Assessment App")
    print("=" * 60)
    
    # Check dependencies
    print("1. Checking dependencies...")
    check_dependencies()
    
    # Check model files
    print("\n2. Checking model files...")
    files_status = check_model_files()
    
    # Determine which app to run
    if files_status['xgb_mental_health_model.json'] and files_status['preprocessor.pkl'] and files_status['label_encoder.pkl']:
        print("\n3. All model files found - using full model integration...")
        app_file = "app_fixed.py"
    elif files_status['xgb_mental_health_model.json']:
        print("\n3. XGBoost model found - using simplified mode...")
        app_file = "app_simple.py"
    else:
        print("\n3. No model files found - using basic mode...")
        app_file = "app_simple.py"
    
    print(f"\n4. Starting application with {app_file}...")
    print("Server will be available at: http://localhost:5000")
    print("Open your browser to access the application")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, app_file])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print(f"\nError starting application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_application()
