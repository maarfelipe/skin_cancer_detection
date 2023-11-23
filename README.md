# Skin Cancer Detection

This repository contains code for a skin cancer detection project. The project includes a machine learning model for classifying skin lesions into different categories, and a Flask web application for running the model on new images.

## Overview

## Data

https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

### Project Structure

- **app.py**: Flask application for running the skin cancer detection model through an API endpoint.
- **skin_cancer_detection.py**: Python script containing the skin cancer detection model definition and loading weights.
- **best_model.h5**: Saved weights of the best-performing model during training.
- **requirements.txt**: List of Python dependencies for running the project.

### Machine Learning Model

The machine learning model is a Convolutional Neural Network (CNN) designed for image classification. It is trained to classify skin lesions into the following categories:

0. Actinic Keratoses and Intraepithelial Carcinomae (Cancer)
1. Basal Cell Carcinoma (Cancer)
2. Benign Keratosis-Like Lesions (Non-Cancerous)
3. Dermatofibroma (Non-Cancerous)
4. Melanocytic Nevi (Non-Cancerous)
5. Pyogenic Granulomas and Hemorrhage (Can lead to cancer)
6. Melanoma (Cancer)

### Flask Web Application

The Flask web application provides an API endpoint ("/api/runmodel") that accepts skin lesion images and returns the predicted category along with informative details about the predicted class.

## Getting Started

1. Install dependencies using `pip install -r requirements.txt`.
2. Run the Flask application using `python app.py`.

## Usage

1. Send a POST request to `http://localhost:5000/api/runmodel` with an image file attached as "pic".
2. The API will return the predicted skin lesion category and informative details.

## Example Usage

```python
import requests
from PIL import Image

url = "http://localhost:5000/api/runmodel"
files = {"pic": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)

data = response.json()
print(data)