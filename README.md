# 🧠 MindCare - Mental Health Assessment Platform

A beautiful, modern, and comprehensive mental health assessment application that uses machine learning to provide personalized mental health insights and recommendations.

## ✨ Features

### 🎨 **Stunning Modern Design**
- **Glass Morphism UI**: Beautiful frosted glass effects with backdrop blur
- **Gradient Backgrounds**: Dynamic purple-blue gradients with floating animations
- **Smooth Animations**: Elegant transitions and hover effects
- **Responsive Design**: Perfect on desktop, tablet, and mobile devices
- **Progress Tracking**: Real-time form completion progress

### 🧠 **Advanced AI Integration**
- **XGBoost Model**: Your trained machine learning model for accurate predictions
- **Real-time Predictions**: Instant mental health assessments
- **Confidence Scores**: Shows prediction confidence percentages
- **Personalized Recommendations**: AI-driven suggestions based on results

### 💙 **Mental Health Focus**
- **Stigma-free Design**: Encouraging, supportive messaging throughout
- **Educational Content**: Helps users understand mental health better
- **Professional Recommendations**: Evidence-based suggestions for care
- **Supportive Language**: Emphasizes seeking help as strength

## 🚀 Quick Start

### **Option 1: Easy Start (Recommended)**
```bash
python run_app.py
```

### **Option 2: Manual Start**
```bash
# Install dependencies
pip install Flask Flask-CORS pandas numpy xgboost scikit-learn

# Start the application
python app.py
```

### **Access the Application**
Open your browser and go to: **http://localhost:5000**

## 📋 Prerequisites

### **Required Files**
Make sure you have these files in your directory:
- `xgb_mental_health_model.json` - Your trained XGBoost model
- `preprocessor.pkl` - Data preprocessing pipeline
- `label_encoder.pkl` - Label encoder for predictions
- `Mental Health Dataset.csv` - Your training dataset

### **Python Dependencies**
- Python 3.8+
- Flask 3.1.2
- Flask-CORS 6.0.1
- pandas
- numpy
- xgboost
- scikit-learn

## 🎯 How It Works

### **1. User Interface**
- **Beautiful Form**: Step-by-step mental health assessment
- **Interactive Elements**: Smooth radio buttons and dropdowns
- **Progress Tracking**: Visual progress bar at the top
- **Real-time Validation**: Form validation as you type

### **2. Data Processing**
- **16 Input Features**: Comprehensive mental health indicators
- **Smart Preprocessing**: Handles all data types and formats
- **Feature Engineering**: Optimized for your trained model

### **3. AI Prediction**
- **XGBoost Model**: Uses your trained machine learning model
- **Confidence Scoring**: Shows how confident the model is
- **Three Outcomes**: Yes, No, or Not sure for mental health care needs

### **4. Personalized Results**
- **Custom Recommendations**: Tailored advice based on prediction
- **Action Items**: Specific steps users can take
- **Support Resources**: Links to mental health resources

## 🎨 Design Highlights

### **Visual Elements**
- **Color Scheme**: Calming purple-blue gradients
- **Typography**: Modern Poppins font family
- **Icons**: Font Awesome icons for visual clarity
- **Animations**: Smooth CSS transitions and keyframe animations

### **User Experience**
- **Intuitive Navigation**: Clear section organization
- **Visual Feedback**: Hover effects and state changes
- **Loading States**: Beautiful spinner animations
- **Result Display**: Engaging result cards with gradients

### **Responsive Design**
- **Mobile First**: Optimized for mobile devices
- **Tablet Friendly**: Perfect layout for tablets
- **Desktop Enhanced**: Rich experience on larger screens
- **Touch Optimized**: Easy interaction on all devices

## 📊 Model Integration

### **Input Features**
Your model analyzes these 16 key factors:
- **Demographics**: Gender, country, occupation
- **Health History**: Family and personal mental health history
- **Lifestyle**: Days indoors, stress levels, habit changes
- **Mental Health Indicators**: Mood swings, coping strategies, social factors

### **Prediction Output**
- **Yes**: Consider professional mental health support
- **Not sure**: Monitor well-being and consider check-ups
- **No**: Continue current healthy practices

### **Confidence Scoring**
- Shows prediction confidence as a percentage
- Helps users understand the reliability of results
- Guides decision-making about next steps

## 🔧 Technical Details

### **Backend (Flask)**
- **RESTful API**: Clean, efficient endpoints
- **Error Handling**: Comprehensive error management
- **CORS Support**: Cross-origin request handling
- **Model Loading**: Automatic model initialization
- **Health Checks**: System status monitoring

### **Frontend (HTML/CSS/JS)**
- **Modern CSS**: Flexbox, Grid, and animations
- **Vanilla JavaScript**: No framework dependencies
- **Progressive Enhancement**: Works without JavaScript
- **Accessibility**: Screen reader friendly

### **API Endpoints**
- `GET /` - Main application interface
- `POST /predict` - Mental health prediction
- `GET /health` - System health check
- `GET /model-info` - Model information

## 🎯 Mental Health Focus

### **Addressing Stigma**
- **Positive Messaging**: Encouraging language throughout
- **Strength-based Approach**: Emphasizes seeking help as strength
- **Supportive Tone**: Non-judgmental, caring interface
- **Educational Content**: Helps users understand mental health

### **User Journey**
1. **Welcoming Introduction**: Sets a supportive tone
2. **Comprehensive Assessment**: Covers all relevant factors
3. **Instant Results**: Immediate feedback with confidence
4. **Actionable Recommendations**: Clear next steps provided

## 📱 Mobile Experience

### **Responsive Features**
- **Touch-friendly**: Large buttons and touch targets
- **Swipe Navigation**: Smooth scrolling and navigation
- **Optimized Layout**: Perfect for mobile screens
- **Fast Loading**: Optimized for mobile networks

## 🚀 Deployment

### **Local Development**
```bash
python run_app.py
```

### **Production Deployment**
1. Set up a production WSGI server (Gunicorn)
2. Configure reverse proxy (Nginx)
3. Set environment variables
4. Deploy to cloud platform (Heroku, AWS, etc.)

## 📈 Future Enhancements

- **User Accounts**: Save assessment history
- **Progress Tracking**: Monitor mental health over time
- **Resource Library**: Mental health resources and articles
- **Community Features**: Support groups and forums
- **Professional Directory**: Connect with mental health professionals

## 🤝 Contributing

This is a mental health-focused application. Please ensure all contributions maintain:
- **Supportive messaging**
- **Stigma-free language**
- **Accessibility standards**
- **Professional medical disclaimers**

## 📞 Support

### **Technical Issues**
- Check the console for error messages
- Verify your model files are properly loaded
- Test on multiple devices and browsers

### **Model Questions**
- Ensure all model files are in the correct directory
- Check that the model was trained with the same features
- Verify the preprocessing pipeline matches your training data

## ⚠️ Important Notes

- **This tool is for informational purposes only**
- **Not a substitute for professional medical advice**
- **Always consult healthcare professionals for mental health concerns**
- **Include appropriate disclaimers in production deployment**

## 🎉 Getting Started

1. **Ensure all model files are present**
2. **Run `python run_app.py`**
3. **Open http://localhost:5000 in your browser**
4. **Fill out the assessment form**
5. **Get your personalized mental health insights!**

---

**Remember**: Mental health matters, and seeking help is a sign of strength! 💙

## 📁 File Structure

```
MODELTRAINER/
├── app.py                          # Flask backend application
├── run_app.py                      # Easy startup script
├── test_app.py                     # Application testing script
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── templates/
│   └── index.html                 # Beautiful frontend interface
├── xgb_mental_health_model.json   # Your trained XGBoost model
├── preprocessor.pkl               # Data preprocessing pipeline
├── label_encoder.pkl              # Label encoder for predictions
└── Mental Health Dataset.csv      # Your training dataset
```

## 🔍 Testing

Run the test script to verify everything works:
```bash
python test_app.py
```

This will test all endpoints and ensure the model is working correctly.#   m e n t a l - h e a l t h - p r e d i c t o r  
 