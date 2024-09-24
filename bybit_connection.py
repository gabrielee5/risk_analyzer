import ccxt
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_bybit_connection():
    exchange_class = ccxt.bybit

    exchange_params = {
        'apiKey': os.getenv('api_key'),
        'secret': os.getenv('api_secret'),
        'enableRateLimit': True,
    }

    exchange = exchange_class(exchange_params)

    return exchange

def test_connection(exchange):
    try:
        # Fetch the balance to test the connection
        balance = exchange.fetch_balance()
        print("Connection successful!")
        print(f"Total BTC balance: {balance['total']['USDT']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    bybit = create_bybit_connection()
    test_connection(bybit)