from main import load_account_credentials
from pybit.unified_trading import HTTP
import time
from dotenv import dotenv_values

def get_account_type(session):
    try:
        # Try to access unified account endpoint
        session.get_wallet_balance(accountType="UNIFIED")
        return "Unified"
    except:
        # If it fails, assume it's a standard account
        return "Standard"

def get_account_balance(session, account_type):
    if account_type == "Unified":
        balance = session.get_wallet_balance(accountType="UNIFIED")
    else:
        balance = session.get_wallet_balance(accountType="CONTRACT")
    
    return balance

def analyze_account(api_key, api_secret):
    session = HTTP(
        api_key=api_key, 
        api_secret=api_secret
        )

    account_type = get_account_type(session)
    print(f"Account Type: {account_type}")
    
    balance = get_account_balance(session, account_type)

    print("\nAccount Information:")
    if 'result' in balance and 'list' in balance['result']:
        for account in balance['result']['list']:
            print(f"Account Type: {account['accountType']}")
            
            print("\nCoin Balances:")
            for coin_data in account['coin']:
                print(f"Coin: {coin_data['coin']}")
                print(f"  Equity: {coin_data['equity']}")
                print(f"  Wallet Balance: {coin_data['walletBalance']}")
                print(f"  Available to Withdraw: {coin_data['availableToWithdraw']}")
                print(f"  Unrealized PnL: {coin_data['unrealisedPnl']}")
                print(f"  Cumulative Realized PnL: {coin_data['cumRealisedPnl']}")
                print()
    else:
        print("Unable to retrieve balance information.")

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
    
    analyze_account(api_key, api_secret)
    

if __name__ == "__main__":
    main()