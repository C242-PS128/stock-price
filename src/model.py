import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def create_model(input_shape):
    """ Create and compile the LSTM model """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
