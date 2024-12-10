import numpy as np
from keras.models import load_model
from src.data_preprocessing import load_current_price_data, load_stock_data, preprocess_data, create_dataset
from src.model import create_model
from sklearn.model_selection import train_test_split

def train_model(stock_code, time_step=60, epochs=50, batch_size=32):
    """ Train the LSTM model """
    # Load & preprocess data
    data = load_stock_data(stock_code)
    current_data = load_current_price_data(stock_code)
    if data is None:
        return None

    scaled_data, scaler = preprocess_data(data, current_data)

    # Create dataset
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # for LSTM

    # Split dataset to (train & test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # (Create & train) model
    model = create_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Save model as "models/{stock_name}.h5"
    save_format = f"models/{stock_code}.h5"
    model.save(save_format)
    return scaler
