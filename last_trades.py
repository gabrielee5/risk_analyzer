from main import load_account_credentials
from pybit.unified_trading import HTTP
import time
from dotenv import dotenv_values
from datetime import datetime
from tabulate import tabulate

def fetch_last_trades(api_key, api_secret):
    session = HTTP(
        api_key=api_key, 
        api_secret=api_secret
        )

    trades = session.get_executions(
    category="linear",
    execType="Trade",
    limit=100,
    )
    return trades

def format_trades(data):
    # Extract the list of trades
    trades = data['result']['list']
    
    # Filter trades with execType 'Trade'
    trade_executions = [trade for trade in trades if trade['execType'] == 'Trade']
    
    # Prepare data for tabulation
    table_data = []
    headers = ["Symbol", "Time", "Side", "Price", "Quantity", "Value", "Fee", "Order Type", "Is Maker"]
    
    for trade in trade_executions:
        # Convert timestamp to readable format
        timestamp = datetime.fromtimestamp(int(trade['execTime']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        
        table_data.append([
            trade['symbol'],
            timestamp,
            trade['side'],
            trade['execPrice'],
            trade['execQty'],
            trade['execValue'],
            trade['execFee'],
            trade['orderType'],
            trade['isMaker']
        ])
    
    # Print the table
    print("Trades with execType 'Trade':")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    # Load all account credentials
    account_credentials = load_account_credentials()
    
    # Print available accounts
    print("Available accounts:")
    for account_id, info in account_credentials.items():
        print(f"ID: {account_id}, Name: {info.get('name', 'Unknown')}")
    
    # Prompt user for account ID
    account_id = input("Enter the account ID to analyze: ")
    
    if account_id not in account_credentials:
        print(f"Error: No credentials found for account ID: {account_id}")
        return
    
    selected_account = account_credentials[account_id]
    selected_account_name = selected_account.get('name', 'Unknown')
    
    print(f"\nAnalyzing account: {selected_account_name} (ID: {account_id})")
    
    api_key = selected_account.get('api_key')
    api_secret = selected_account.get('api_secret')
    
    if not api_key or not api_secret:
        print(f"Error: Missing API credentials for account ID: {account_id}")
        return
    
    trades = fetch_last_trades(api_key, api_secret)
    format_trades(trades)


if __name__ == "__main__":
    main()