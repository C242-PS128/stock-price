import numpy as np
from keras.models import load_model
from src.data_preprocessing import load_current_price_data, load_stock_data, preprocess_data, create_dataset
from src.model import create_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

import joblib

def train_model(stock_code, time_step=60, epochs=50, batch_size=32, patience=5):
    """ Train the LSTM model """
    # Load & preprocess data
    data = load_stock_data(stock_code)
    current_data = load_current_price_data(stock_code)
    if data is None:
        return None

    scaled_data, scaler = preprocess_data(data, current_data)

    # Create dataset
    X, y = create_dataset(scaled_data, time_step)
    
    # Check if the dataset can be reshaped
    if X.shape[0] < 1:
        print("Not enough data to create the dataset.")
        return None

    # Adjust the dataset size if necessary
    num_samples = X.shape[0]
    if num_samples % time_step != 0:
        # Calculate the number of samples that can be reshaped
        new_size = num_samples - (num_samples % time_step)
        X = X[:new_size]
        y = y[:new_size]

    # Reshape for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)  # for LSTM

    # Split dataset to (train & test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # (Create & train) model
    model = create_model((X_train.shape[1], 1))
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Train the model with early stopping
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Save model as "models/{stock_name}.h5"
    save_format = f"models/{stock_code}.h5"
    model.save(save_format)

    # Save the scaler
    joblib.dump(scaler, f"models/{stock_code}_scaler.pkl")  # Save the scaler

    return scaler
