import matplotlib.pyplot as plt

def plot_results(actual, predicted):
    """ Plot actual vs predicted stock prices """
    plt.figure(figsize=(14, 5))
    plt.plot(actual, color='blue', label='Actual Stock Price')
    plt.plot(predicted, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
