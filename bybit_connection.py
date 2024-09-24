import ccxt
from dotenv import dotenv_values
import os

def create_bybit_connection():
    exchange_class = ccxt.bybit

    # Load environment variables
    secrets = dotenv_values(".env")
    api_key = secrets["api_key"]
    api_secret = secrets["api_secret"]

    exchange_params = {
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    }

    exchange = exchange_class(exchange_params)

    return exchange

def test_connection(exchange):
    try:
        # Fetch the balance to test the connection
        balance = exchange.fetch_balance()
        print("Connection successful!")
        print(f"Total USDT balance: {balance['total']['USDT']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    bybit = create_bybit_connection()
    test_connection(bybit)