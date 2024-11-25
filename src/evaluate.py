import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_preprocessing import load_stock_data, preprocess_data, create_dataset

def evaluate_model(stock_code, scaler, time_step=60):
    """ Evaluate the trained LSTM model """
    # Load & preprocess data
    data = load_stock_data(stock_code)
    if data is None:
        return None

    scaled_data, _ = preprocess_data(data)

    # Create dataset
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # for LSTM

    # Load model
    model = load_model('lstm_model.h5')

    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)  # Inverse scaling
    y_inverse = scaler.inverse_transform(y.reshape(-1, 1))

    # Calculate metrics
    mae = mean_absolute_error(y_inverse, predictions)
    mse = mean_squared_error(y_inverse, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_inverse, predictions)

    # Prepare results
    results = pd.DataFrame({'Actual': y_inverse.flatten(), 'Predicted': predictions.flatten()})
    results_metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }
    
    return results, results_metrics
