# Stock Price Prediction

This project implements a machine learning model for predicting stock prices using Python. It utilizes various libraries and frameworks to train, evaluate, and deploy the model via a RESTful API.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Requirements](#requirements)

## Installation

1. Clone the repository
2. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train and evaluate the models, run the following command:
```bash
python main.py    
```
You can choose which model to train in the main.py file by changing the stock code. This will train models for the specified stock codes and print evaluation metrics.

To start the API for predictions and evaluations, run:
```bash
pyton api.py
```

## API Endpoints

### Evaluate Model

- **Endpoint:** `/evaluate`
- **Method:** `POST`
- **Request Body:**
    ```json 
    { 
        "stock_code": "BBCA" 
    }
    ```

- **Response:**
Returns evaluation metrics and results of the model.

### Predict Future Prices

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Request Body:**
    ```json 
    { 
        "stock_code": "BBCA", 
        "start_date": "2024-01-01", 
        "end_date": "2024-01-10" 
    }
- **Response:**
  - Returns predicted stock prices for the specified date range.

## Requirements

- Python 3.x
- Flask
- Keras
- scikit-learn
- pandas
- joblib
- numpy

For a complete list of dependencies, refer to the `requirements.txt` file.