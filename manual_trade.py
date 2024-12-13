from pybit.unified_trading import HTTP
from dotenv import dotenv_values

def initialize_account(account):
    secrets = dotenv_values(".env")
    api_key = secrets[f"{account}_api_key"]
    api_secret = secrets[f"{account}_api_secret"]

    session = HTTP(
            api_key=api_key, 
            api_secret=api_secret
            )
    return session

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
    account = "003"
    symbol = "BTCUSDT"
    side = "Buy"
    qty = 0.001
    
    confirmation = input(f"ID: {account} - About to {side} {qty} {symbol}. Confirm? [y] ")

    if confirmation == "y":
        session = initialize_account(account)
        trade = manual_trade(session, symbol, side, qty)
        print(trade)