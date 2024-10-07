import ccxt
from dotenv import dotenv_values
import os
import logging

# not in use anymore; keeping it to test other funcs

def create_bybit_connection():
    exchange_class = ccxt.bybit

    # Load environment variables
    secrets = dotenv_values(".env")
    api_key = secrets["001_api_key"]
    api_secret = secrets["001_api_secret"]

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

# added to test functions
def fetch_account_data(exchange):
    balance = exchange.fetch_balance()
    prova = exchange.fetchBalance()
    portfolio_value = balance['total']['USDT']
    positions = exchange.fetchPositions()
    print(f"Portfolio Value: ${portfolio_value:.2f}")
    print(f"Portfolio Positions: {positions}")
    # print(prova)

def calculate_max_loss(exchange):
    positions = exchange.fetchPositions()
    total_max_loss = 0
    total_max_loss_long = 0
    total_max_loss_short = 0

    print("Maximum Loss for Each Position:")
    print("--------------------------------")
    
    for position in positions:
        symbol = position['symbol']
        side = position['side']
        mark_price = position['markPrice']
        contracts = position['contracts']
        liquidation_price = position['liquidationPrice']
        notional = position['notional']
        
        if side == 'long':
            max_loss = notional - (contracts * liquidation_price)
            max_loss_2 = (mark_price - liquidation_price) * contracts
            total_max_loss_long += max_loss_2
        elif side == 'short':
            max_loss = (contracts * liquidation_price) - notional
            max_loss_2 = (liquidation_price - mark_price) * contracts
            total_max_loss_short += max_loss_2
        else:
            print(f"Unknown position side for {symbol}: {side}")
            continue
        
        print(f"{symbol} ({side}): ${max_loss_2:.2f}")
        total_max_loss += max_loss_2
    
    print("\nTotal Maximum Loss: ${:.2f}".format(total_max_loss))
    print("\nTotal Maximum Loss Long: ${:.2f}".format(total_max_loss_long))
    print("\nTotal Maximum Loss Short: ${:.2f}".format(total_max_loss_short))

def fetch_positions(exchange):
    positions = exchange.fetchPositions()
    print(positions)

if __name__ == "__main__":
    bybit = create_bybit_connection()
    # test_connection(bybit)
    fetch_positions(bybit)