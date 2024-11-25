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
        stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])
        stock_data.set_index('timestamp', inplace=True)
        return stock_data
    else:
        print(f"File for stock code {stock_code} not found.")
        return None

def preprocess_data(data):
    """ Preprocess the data for LSTM """
    data = data[['close']]
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

def create_dataset(data, time_step=1):
    """ Create dataset for LSTM """
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
