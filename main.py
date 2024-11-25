from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import plot_results

if __name__ == "__main__":
    # Stock code to train and evaluate
    stock_code = 'ABBA'  # Code (can be seen at DaftarSaham.csv file)

    # Train the model
    scaler = train_model(stock_code)

    # Evaluate the model
    results, metrics = evaluate_model(stock_code, scaler)

    # Print evaluation metrics
    if results is not None:
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Plot results
        plot_results(results['Actual'], results['Predicted'])
