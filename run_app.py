#!/usr/bin/env python3
"""
MindCare Mental Health Assessment App - Startup Script
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
        print(f"Missing packages: {missing_packages}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Dependencies installed successfully!")
    else:
        print("All dependencies are installed!")

def check_model_files():
    """Check if all required model files exist"""
    required_files = [
        'xgb_mental_health_model.json',
        'preprocessor.pkl',
        'label_encoder.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing model files: {missing_files}")
        print("Please ensure all model files are in the current directory.")
        return False
    else:
        print("All model files found!")
        return True

def start_application():
    """Start the Flask application"""
    print("=" * 60)
    print("🧠 MindCare - Mental Health Assessment App")
    print("=" * 60)
    
    # Check dependencies
    print("1. Checking dependencies...")
    check_dependencies()
    
    # Check model files
    print("\n2. Checking model files...")
    if not check_model_files():
        print("❌ Cannot start application without model files.")
        return False
    
    # Start the application
    print("\n3. Starting Flask application...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Open your browser to access the application")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_application()
