from pybit.unified_trading import HTTP
import time
from dotenv import dotenv_values

def get_account_credentials():
    secrets = dotenv_values(".env")
    accounts = {}
    for i in range(1, 1000):  # Assuming a maximum of 999 accounts
        prefix = f"{i:03d}_"  # Creates padded numbers like 001, 002, etc.
        api_key = secrets.get(f"{prefix}api_key")
        api_secret = secrets.get(f"{prefix}api_secret")
        name = secrets.get(f"{prefix}name")
        
        if api_key and api_secret and name:
            accounts[prefix[:-1]] = {
                "api_key": api_key,
                "api_secret": api_secret,
                "name": name
            }
        else:
            break  # Stop if we don't find a complete set of credentials
    return accounts

def choose_account(accounts):
    print("Available accounts:")
    for account_id, details in accounts.items():
        print(f"{account_id}: {details['name']}")
    while True:
        choice = input("Enter the account ID you want to use: ")
        if choice in accounts:
            return choice
        else:
            print("Invalid account ID. Please try again.")

def create_session(api_key, api_secret):
    return HTTP(api_key=api_key, api_secret=api_secret)

def cancel_all_orders(session):
    response = session.cancel_all_orders(
        category="linear",
        settleCoin="USDT",
    )
    return response

def get_perp_positions(session):
    positions_info = {}
    response = session.get_positions(
        category="linear",
        symbol="",
        settleCoin="USDT"
    )
    # Ensure the response is successful
    if response.get('retCode') == 0:
        positions_list = response.get('result', {}).get('list', [])

        for position in positions_list:
            symbol = position.get('symbol')
            position_details = {
                'size': position.get('size'),
                'side': position.get('side')}
            positions_info[symbol] = position_details

    if positions_info == {}:
        print("No open position found.")
    else:
        print(f"Found {len(positions_info)} open positions.")
    
    return positions_info

def close_perp_position(session, symbol, qty, side):
    opposite_side = "Sell" if side == "Buy" else "Buy"
    response = session.place_order(
    category="linear",
    symbol=symbol,
    side=opposite_side,
    orderType="Market",
    qty=qty,
    reduceOnly=True,
    )
    return response

def close_all_positions(session, positions):
    if positions is None:
        print("There are no open perpetual positions.")
    else:
        print("Found open perpetual positions: ", positions)
        for symbol, details in positions.items():
            close_perp_position(session, symbol, details['size'], details['side'])
            print(f"Closed {symbol}.")

def main():
    accounts = get_account_credentials()
    if not accounts:
        print("No accounts found in .env file.")
        return
    
    chosen_account_id = choose_account(accounts)
    api_key = accounts[chosen_account_id]["api_key"]
    api_secret = accounts[chosen_account_id]["api_secret"]
    account_name = accounts[chosen_account_id]["name"]
    
    print(f"Using account: {account_name} (ID: {chosen_account_id})")
    session = create_session(api_key, api_secret)

    confirmation = input(f"Are you sure you want to close everything for {account_name} (ID: {chosen_account_id})? [yes] ")
    if confirmation == "yes":
        cancel_all_orders(session)
        positions = get_perp_positions(session)
        close_all_positions(session, positions)

        print("Process completed. Checking if everything is closed...")
        get_perp_positions(session)
    else:
        print("Aborting.")

if __name__ == "__main__":
    main()