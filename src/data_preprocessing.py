import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

def load_data(file_path):
    """ Load stock price data from a CSV file """
    data = pd.read_csv(file_path)
    return data

def load_stock_data(stock_code):
    """ Load daily stock data for a given stock code """
    file_path = f'data/raw/daily/{stock_code}.csv'
    print(file_path)
    if os.path.exists(file_path):
        stock_data = pd.read_csv(file_path)
        stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp']).dt.tz_localize(None)
        stock_data.set_index('timestamp', inplace=True)
        return stock_data
    else:
        print(f"File for stock code {stock_code} not found.")
        return None

def load_current_price_data(stock_code):
    """ Load the last 100 days of current price data for a given stock code """
    file_path = f'data/current_price/{stock_code}.csv'
    if os.path.exists(file_path):
        current_data = pd.read_csv(file_path)
        current_data['date'] = pd.to_datetime(current_data['date']).dt.tz_localize(None)
        current_data.set_index('date', inplace=True)
        current_data = current_data[['open', 'high', 'low', 'close', 'volume']]
        return current_data
    else:
        print(f"Current price data for stock code {stock_code} not found.")
        return None

def preprocess_data(data, current_data=None):
    """ Preprocess the data for LSTM """
    if current_data is not None:
        # Combine datasets first
        combined_data = pd.concat([data, current_data])
        
        # Sort the combined dataset by date
        combined_data.sort_index(inplace=True)

        # Create a complete date range from the minimum to maximum date
        full_date_range = pd.date_range(start=combined_data.index.min(), end=combined_data.index.max(), freq='D')
        
        # Reindex the combined data to the full date range
        combined_data = combined_data.reindex(full_date_range)

        # Fill missing values using linear interpolation
        combined_data.interpolate(method='linear', inplace=True)
    else:
        combined_data = data

    combined_data = combined_data[['close']]
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data)

    return scaled_data, scaler

def create_dataset(data, time_step=1):
    """ Create dataset for LSTM """
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
