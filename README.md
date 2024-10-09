
# Bank Note Authentication with FastAPI

This repository contains a FastAPI application for authenticating bank notes using machine learning.

## Project Overview
This project predicts the authenticity of bank notes based on features such as variance, skewness, kurtosis, and entropy. It uses a pre-trained machine learning model to classify whether a note is genuine or counterfeit.

## Files
- `BankNote_Authentication.csv`: Dataset used for model training.
- `ModelTraining.ipynb`: Jupyter notebook for training the model.
- `main.py`: FastAPI app for predictions.
- `classifier.pkl`: Serialized trained model.
- `requirements.txt`: Project dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the FastAPI app:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints
- `/predict`: Accepts input features (variance, skewness, kurtosis, entropy) and returns a prediction on whether the banknote is authentic or not.
