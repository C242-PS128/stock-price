from concurrent.futures import ProcessPoolExecutor, as_completed
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import plot_results

def train_and_evaluate(stock_code):
    """ Train and evaluate the model for a given stock code """
    print(f"Training model for {stock_code}...")
    # Train the model
    scaler = train_model(stock_code)

    # Evaluate the model
    results, metrics = evaluate_model(stock_code, scaler)

    return stock_code, results, metrics

if __name__ == "__main__":
    # Chosen stock codes
    stock_codes = ['BBCA', 'BBRI', 'BMRI', 'INKP', 'TKIM', 'BYAN', 
                'TMAS', 'ASII', 'TLKM', 'UNVR', 'AMRT', 'ADRO']

    # Use ProcessPoolExecutor to run training in parallel
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(train_and_evaluate, stock_code): stock_code for stock_code in stock_codes}

        for future in as_completed(futures):
            stock_code = futures[future]
            try:
                stock_code, results, metrics = future.result()
                # Print evaluation metrics
                if results is not None:
                    print(f"Evaluation Metrics for {stock_code}:")
                    for key, value in metrics.items():
                        print(f"{key}: {value:.4f}")

                    # Plot results
                    plot_results(results['Actual'], results['Predicted'])
                else:
                    print(f"Failed to evaluate model for {stock_code}.")
            except Exception as e:
                print(f"Error occurred for {stock_code}: {e}")
