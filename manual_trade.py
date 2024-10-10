from pybit.unified_trading import HTTP
from dotenv import dotenv_values

account = "003"
secrets = dotenv_values(".env")
api_key = secrets[f"{account}_api_key"]
api_secret = secrets[f"{account}_api_secret"]

session = HTTP(
        api_key=api_key, 
        api_secret=api_secret
        )

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
    symbol = "BTCUSDT"
    side = "Buy"
    qty = 0.001
    confirmation = input(f"ID: {account} - About to {side} {qty} {symbol}. Confirm? [y] ")
    if confirmation == "y":
        trade = manual_trade(session, symbol, side, qty)
        print(trade)