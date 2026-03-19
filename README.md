---
library_name: scikit-learn
tags:
- regression
- food-waste
- sustainability
- scikit-learn
---



# FoodBridge AI
## Intelligent Global Food Waste Redistribution System

FoodBridge AI is a machine learning system designed to predict food surplus and identify waste levels in food service environments. The system helps reduce food waste by enabling smarter redistribution strategies.

---

# Project Objectives

The goal of this project is to:

- Predict food surplus using machine learning
- Classify waste levels (Low / Medium / High)
- Provide real-time predictions through APIs
- Offer an interactive dashboard for users
- Support data-driven food redistribution

---

# Project Architecture

FoodBridge AI consists of the following components:

1. **Exploratory Data Analysis (EDA)**
2. **Machine Learning Models**
3. **Model Training Pipelines**
4. **FastAPI Backend**
5. **Streamlit Dashboard**
6. **Model Documentation**

---

# Project Structure

# FoodBridge AI
## Intelligent Global Food Waste Redistribution System

FoodBridge AI is a machine learning system designed to predict food surplus and identify waste levels in food service environments. The system helps reduce food waste by enabling smarter redistribution strategies.

---

# Project Objectives

The goal of this project is to:

- Predict food surplus using machine learning
- Classify waste levels (Low / Medium / High)
- Provide real-time predictions through APIs
- Offer an interactive dashboard for users
- Support data-driven food redistribution

---

# Project Architecture

FoodBridge AI consists of the following components:

1. **Exploratory Data Analysis (EDA)**
2. **Machine Learning Models**
3. **Model Training Pipelines**
4. **FastAPI Backend**
5. **Streamlit Dashboard**
6. **Model Documentation**

---

# Project Structure

FoodBridge_AI
│
├── data
│
├── notebooks
│ ├── 01_EDA.ipynb
│ ├── 02_surplus_prediction_model.ipynb
│ └── 03_waste_classification_model.ipynb
│
├── utils
│
├── models
│
├── api
│ └── main.py
│
├── dashboard
│ └── app.py
│
├── saved_models
│ ├── foodbridge_regressor.pkl
│ ├── waste_classifier.pkl
│ ├── model_features.json
│ └── classifier_features.json
│
├── model_cards
│ └── foodbridge_model_card.md
│
└── README.md


---

# Dataset

The dataset contains approximately **10,000 records** collected from various food service environments.

Key attributes include:

- food preparation quantity
- customer footfall
- demand indicators
- environmental factors
- location information

---

# Machine Learning Models

## Surplus Prediction

Algorithm used:

Random Forest Regressor

Performance:

MAE ≈ 1.7  
R² ≈ 0.99

---

## Waste Classification

Algorithm used:

Random Forest Classifier

Performance:

Accuracy ≈ 90%

---

# Technologies Used

- Python
- Pandas
- Scikit-Learn
- FastAPI
- Streamlit
- Matplotlib
- Seaborn

---

# Running the Project

## 1 Install Dependencies


---

# Dataset

The dataset contains approximately **10,000 records** collected from various food service environments.

Key attributes include:

- food preparation quantity
- customer footfall
- demand indicators
- environmental factors
- location information

---

# Machine Learning Models

## Surplus Prediction

Algorithm used:

Random Forest Regressor

Performance:

MAE ≈ 1.7  
R² ≈ 0.99

---

## Waste Classification

Algorithm used:

Random Forest Classifier

Performance:

Accuracy ≈ 90%

---

# Technologies Used

- Python
- Pandas
- Scikit-Learn
- FastAPI
- Streamlit
- Matplotlib
- Seaborn

---

# Running the Project

## 1 Install Dependencies

    pip install -r requirements.txt
---

## 2 Run FastAPI Server

    cd api
    uvicorn main:app --reload

API documentation will be available at:

    http://127.0.0.1:8000/docs


---

## 3 Run Streamlit Dashboard

    cd dashboard
    streamlit run app.py


Dashboard will open at:

    http://localhost:8501


---

# Applications

FoodBridge AI can be used by:

- restaurants
- supermarkets
- NGOs
- food banks
- smart city initiatives

---

# Future Enhancements

- Real-time IoT food monitoring
- Deep learning demand forecasting
- NGO logistics integration
- Geographic redistribution optimization

---

# Author

Final Project

FoodBridge AI – Intelligent Food Waste Prediction System


