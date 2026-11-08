# Handwritten ID Recognition

A Machine Learning-based application designed to recognize handwritten student IDs from uploaded images.

## Overview

Handwritten ID Recognition is a web application that uses a trained Machine Learning model to classify handwritten student ID images into their corresponding classes.

Users can upload an image of handwritten text through the web interface. The application sends the image to a backend API, which processes it using the trained model and returns the predicted class along with its probability scores.

## Features

- Upload handwritten ID images in PNG or JPG format.
- Predict handwritten ID classes using a trained Machine Learning model.
- Display the predicted class and confidence score.
- Visualize probability scores for each class.
- Check backend API connectivity.

## Tech Stack

**Frontend**
- Python
- Streamlit
- Requests
- Pandas
- Altair

**Backend**
- Python
- FastAPI
- Uvicorn
- TensorFlow / Keras

## Installation and Setup

### Prerequisites

- Python installed
- pip package manager
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd A5-UAS-ML
```

### 2. Set Up the Backend

Open a terminal and navigate to the backend directory.

```bash
cd backend
```

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check backend connectivity |
| POST | `/predict` | Predict handwritten ID from an uploaded image |

### 3. Set Up the Frontend

Open a **separate terminal** and navigate to the frontend directory.

```bash
cd frontend
```

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

Start the Streamlit application.

```bash
streamlit run main.py
```

The frontend will be available at:

```text
http://localhost:8501
```

## Application Workflow

1. Upload an image containing handwritten student ID.
2. The frontend sends the image to the backend through the `/predict` endpoint.
3. The backend processes the image using the trained Machine Learning model.
4. The model returns its predicted class and probability scores.
5. The frontend displays the prediction results and probability visualization.

## API Configuration

The frontend uses the following default prediction API URL:

```text
http://127.0.0.1:8000/predict
```

To configure a different backend address, set the `PREDICT_URL` environment variable.

**Windows PowerShell:**

```powershell
$env:PREDICT_URL = "http://your-backend-address/predict"
```

## Notes

- The backend must be running before making predictions through the frontend.
- The model and its class mapping must use the same class order established during training.
- Predicted numeric class indices should be mapped to their corresponding student IDs before being presented as final identities.

## Project Information

**Project:** Handwritten ID Recognition  
**Course:** Machine Learning  
**Event:** Metta Day 2026  

© 2026 All rights reserved.