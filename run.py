#!/usr/bin/env python3
"""
MindCare Mental Health Assessment App - Simple Startup
"""

import subprocess
import sys
import webbrowser
import time

def start_application():
    """Start the Flask application"""
    print("=" * 60)
    print("MindCare - Mental Health Assessment App")
    print("=" * 60)
    print("Starting application...")
    print("Server will be available at: http://localhost:5000")
    print("Open your browser to access the application")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "app_final.py"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print(f"\nError starting application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_application()
