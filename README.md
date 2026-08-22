# 🌱 AI Crop Disease Detection

An AI-powered web application that detects crop diseases from leaf images using a deep learning model. The system helps farmers identify possible crop diseases and provides useful information for disease prevention and crop management.

## 📌 Project Overview

AI Crop Disease Detection is a Flask-based web application developed using Python and TensorFlow/Keras.

Users can register and log in, upload an image of a crop leaf, and receive an AI-based disease prediction with a confidence score.

The application also provides prediction history, farmer profile management, weather information, and an AI chat assistant for crop-related questions.

## ✨ Features

- 🌿 Crop leaf disease detection using AI
- 📷 Upload crop leaf images
- 🤖 Deep learning-based disease prediction
- 📊 Prediction confidence score
- 📜 Prediction history
- 👨‍🌾 Farmer registration and login
- 👤 Farmer profile
- 🌦️ Weather dashboard
- 💬 AI chat assistant
- 🗄️ Database integration
- 🎨 User-friendly web interface

## 🛠️ Technologies Used

### Programming Languages
- Python
- HTML
- CSS
- JavaScript

### Frameworks & Libraries
- Flask
- TensorFlow
- Keras
- NumPy
- Pandas
- Pillow

### Database
- SQLite

### Machine Learning
- Deep Learning
- Convolutional Neural Network (CNN)
- TensorFlow/Keras

### Development Tools
- Visual Studio Code
- Git
- GitHub

## 📂 Project Structure

```text
AI-Crop-Disease-Detection/
│
├── app.py
├── database.py
├── train_model.py
├── train_model_balanced.py
├── train_model_correct.py
│
├── model/
│   ├── crop_disease_model_correct.keras
│   ├── labels.txt
│   ├── labels_balanced.txt
│   ├── labels_correct.txt
│   └── predict.py
│
├── static/
│   ├── css/
│   │   ├── style.css
│   │   └── dashboard.css
│   └── uploads/
│
├── templates/
│   ├── chat.html
│   ├── dashboard.html
│   ├── history.html
│   ├── index.html
│   ├── login.html
│   ├── profile.html
│   ├── register.html
│   ├── result.html
│   ├── upload.html
│   └── weather.html
│
├── .gitignore
└── README.md
## 📸 Project Screenshots

### 🏠 Home Page
![AgroVision AI Home Page](homepage.png)

### 📊 Farmer Dashboard
![AgroVision AI Dashboard](dashboard.png)
### 🔬 Disease Prediction Result

![AI Crop Disease Detection Result](screenshots/result.png)
