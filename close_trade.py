from pybit.unified_trading import HTTP
from dotenv import dotenv_values
import ccxt

from main import load_account_credentials


def initialize_account(account):
    secrets = dotenv_values(".env")
    api_key = secrets[f"{account}_api_key"]
    api_secret = secrets[f"{account}_api_secret"]
    session = HTTP(
            api_key=api_key, 
            api_secret=api_secret
            )
    return session

def get_position_info(session, symbol):
    """Get position information for a specific symbol"""
    positions = session.get_positions(
        category="linear",
        symbol=symbol
    )
    
    # Extract the position data
    if "result" in positions and "list" in positions["result"]:
        for position in positions["result"]["list"]:
            if position["symbol"] == symbol:
                return position
    return None

def close_position(session, symbol):
    """Close a position for a specific symbol"""
    # Get current position
    position = get_position_info(session, symbol)
    
    if not position or float(position["size"]) == 0:
        print(f"No open position found for {symbol}")
        return None
    
    # Determine side for closing order (opposite of current position)
    position_side = position["side"]
    closing_side = "Sell" if position_side == "Buy" else "Buy"
    
    # Get quantity to close (absolute value of position size)
    qty = abs(float(position["size"]))
    
    # Place closing order
    response = session.place_order(
        category="linear",
        symbol=symbol,
        side=closing_side,
        orderType="Market",
        qty=str(qty),
        reduceOnly=True,
    )
    
    return response

def manual_trade(session, symbol, side, qty):
    response = session.place_order(
        category="linear",
        symbol=symbol,
        side=side,
        orderType="Market",
        qty=qty,
        reduceOnly=True,
        )
    return response

if __name__ == "__main__":

    account_credentials = load_account_credentials()
    print("Available accounts:")
    for account_id, info in account_credentials.items():
        print(f"ID: {account_id}, Name: {info.get('name', 'Unknown')}")

    account = input("Enter account ID (e.g., 001): ")
    symbol = input("Enter symbol to close position (e.g., BTCUSDT): ")
    # symbol = "FORTHUSDT"  

    session = initialize_account(account)
    
    # Get position before closing for confirmation
    position = get_position_info(session, symbol)
    
    if position and float(position["size"]) != 0:
        position_side = position["side"]
        position_size = position["size"]
        closing_side = "Sell" if position_side == "Buy" else "Buy"
        
        confirmation = input(f"ID: {account} - About to close {position_side} position of {position_size} {symbol} by placing a {closing_side} order. Confirm? [y] ")
        
        if confirmation.lower() == "y":
            trade = close_position(session, symbol)
            print(f"Position closing order placed: {trade}")
        else:
            print("Operation cancelled.")
    else:
        print(f"No open position found for {symbol}")