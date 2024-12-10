from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import plot_results

if __name__ == "__main__":
    # Choosen stock codes
    stock_codes = ['BBCA', 'BBRI', 'BMRI', 'INKP', 'TKIM', 'BYAN', 
                   'TMAS', 'ASII', 'TLKM', 'UNVR', 'AMRT', 'ADRO']

    for stock_code in stock_codes:
        print(f"\nTraining model for {stock_code}...")

        # Train the model
        scaler = train_model(stock_code)

        # Evaluate the model
        results, metrics = evaluate_model(stock_code, scaler)

        # Print evaluation metrics
        if results is not None:
            print(f"Evaluation Metrics for {stock_code}:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

            # Plot results
            plot_results(results['Actual'], results['Predicted'])
        else:
            print(f"Failed to evaluate model for {stock_code}.")
