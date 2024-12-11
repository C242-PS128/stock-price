from flask import Flask, request, jsonify
import numpy as np
import tempfile
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_preprocessing import load_stock_data, preprocess_data, create_dataset
from keras.models import load_model
import pandas as pd
import joblib
import os
import requests

app = Flask(__name__)

def download_file(bucket_name, file_name, destination_file_name):
    url = f"https://storage.googleapis.com/{bucket_name}/{file_name}"
    response = requests.get(url)

    if response.status_code == 200:
        with open(destination_file_name, 'wb') as f:
            f.write(response.content)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """ API endpoint to evaluate the model """
    data = request.json
    stock_code = data.get('stock_code')

    if not stock_code:
        return jsonify({'error': 'Please provide stock_code.'}), 400

    BUCKET = 'financebro-bucket'
    MODEL_STOCK = f"models/{stock_code}.h5"
    SCALER_STOCK = f"models/{stock_code}_scaler.pkl"

    with tempfile.TemporaryDirectory() as tmpdirname:
        models_dir = os.path.join(tmpdirname, 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, f"{stock_code}.h5")
        scaler_path = os.path.join(models_dir, f"{stock_code}_scaler.pkl")

        try:
            download_file(BUCKET, MODEL_STOCK, model_path)
            download_file(BUCKET, SCALER_STOCK, scaler_path)

            # Load the model and scaler
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            return jsonify({'error': f'Failed to load model or scaler: {str(e)}'}), 500

    # Load historical data for evaluation
    historical_data = load_stock_data(stock_code)
    if historical_data is None:
        return jsonify({'error': 'Historical data not found.'}), 404

    # Preprocess the data
    scaled_data, _ = preprocess_data(historical_data)

    # Create dataset for evaluation
    time_step = 60  # Use the same time step as during training
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Inverse transform the actual values
    y_inverse = scaler.inverse_transform(y.reshape(-1, 1)) 

    # Convert actual values to integers
    y_inverse = y_inverse.astype(int) 

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_inverse, predictions)
    mse = mean_squared_error(y_inverse, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_inverse, predictions)

    # Prepare results
    results = pd.DataFrame({
        'Actual': y_inverse.flatten(),
        'Predicted': predictions.flatten()
    })

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

    return jsonify({
        'metrics': metrics,
        'results': results.to_dict(orient='index')
    }), 200

    
@app.route('/predict', methods=['POST'])
def predict():
    """ API endpoint to predict future stock prices """
    data = request.json
    stock_code = data.get('stock_code')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    if not stock_code or not start_date or not end_date:
        return jsonify({'error': 'Please provide stock_code, start_date, and end_date.'}), 400

    BUCKET = 'financebro-bucket'
    MODEL_STOCK = f"models/{stock_code}.h5"
    SCALER_STOCK = f"models/{stock_code}_scaler.pkl"

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create the models directory inside the temporary directory
        models_dir = os.path.join(tmpdirname, 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Define paths for temporary files
        model_path = os.path.join(models_dir, f"{stock_code}.h5")
        scaler_path = os.path.join(models_dir, f"{stock_code}_scaler.pkl")

        # Download the model and scaler from GCS
        try:
            download_file(BUCKET, MODEL_STOCK, model_path)
            download_file(BUCKET, SCALER_STOCK, scaler_path)

            # Load the model and scaler
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            return jsonify({'error': f'Failed to load model or scaler: {str(e)}'}), 500

    # Load historical data for prediction
    historical_data = load_stock_data(stock_code)
    if historical_data is None:
        return jsonify({'error': 'Historical data not found.'}), 404

    # Preprocess the data
    scaled_data, _ = preprocess_data(historical_data)

    # Create dataset for prediction
    time_step = 100
    X, _ = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1) 

    # Make predictions for the next days
    predictions = []
    last_sequence = X[-1]

    # Predict future prices
    for _ in range((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1):
        pred = model.predict(last_sequence.reshape(1, time_step, 1))
        predictions.append(pred[0, 0])  # Store the predicted price
        # Update the last sequence with the new prediction
        last_sequence = np.append(last_sequence[1:], pred)

    # Inverse transform the predictions to original price
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Prepare a DataFrame for the predictions
    prediction_dates = pd.date_range(start=start_date, end=end_date)
    prediction_df = pd.DataFrame(data=predictions, index=prediction_dates, columns=['Predicted Price'])

    # Convert index to string for JSON serialization
    prediction_df.index = prediction_df.index.astype(str)

    return jsonify(prediction_df.to_dict(orient='index')), 200
    
if __name__ == "__main__":
    app.run(debug=True)
