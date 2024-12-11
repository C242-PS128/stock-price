import requests
import pandas as pd
from datetime import datetime
import os
import time

def fetch_stock_data(symbol):
    """
    Fetch stock data from MarketStack API for a specific symbol
    """
    base_url = "https://api.marketstack.com/v1/eod"
    access_key = "50d9fd88a487330032b65fd99934488d"
    
    params = {
        'access_key': access_key,
        'symbols': f"{symbol}.XIDX"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def process_stock_data(data):
    """
    Process the MarketStack API response and convert it to a pandas DataFrame
    """
    if not data or 'data' not in data:
        return None
        
    df = pd.DataFrame(data['data'])
    return df

def save_to_csv(df, symbol):
    """
    Save the DataFrame to a CSV file
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"data/current_price/{symbol}_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():
    # List of stocks
    stocks = ['BBCA', 'BBRI', 'BMRI', 'INKP', 'TKIM', 'BYAN', 
              'TMAS', 'ASII', 'TLKM', 'UNVR', 'AMRT', 'ADRO']
    
    # Process each stock
    for stock in stocks:
        print(f"\nProcessing {stock}...")
        
        # Fetch data
        raw_data = fetch_stock_data(stock)
        
        if raw_data:
            # Process data
            df = process_stock_data(raw_data)
            
            if df is not None:
                # Save to CSV
                save_to_csv(df, stock)
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(1)
            else:
                print(f"No data available for {stock}")
        else:
            print(f"Skipping {stock} due to error in data fetch")

if __name__ == "__main__":
    main()
