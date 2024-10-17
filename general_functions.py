from dotenv import load_dotenv
import re
import os
from pybit.unified_trading import HTTP
from tabulate import tabulate

# AUTH
def load_account_credentials():
    load_dotenv()
    
    account_credentials = {}
    account_pattern = re.compile(r'(\d+)_(\w+)')
    
    for key, value in os.environ.items():
        match = account_pattern.match(key)
        if match:
            account_id, credential_type = match.groups()
            if account_id not in account_credentials:
                account_credentials[account_id] = {}
            account_credentials[account_id][credential_type] = value
    
    return account_credentials

def get_account_keys():
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
    
    return api_key, api_secret

def establish_session(api_key, api_secret):
    session = HTTP(
        api_key=api_key, 
        api_secret=api_secret
        )
    return session

def get_account_type(session):
    try:
        # Try to access unified account endpoint
        session.get_wallet_balance(accountType="UNIFIED")
        return "UNIFIED"
    except:
        # If it fails, assume it's a standard account
        return "CONTRACT"

# FORMATTING
def create_table(data):
    if data:
        table = [[key, value] for key, value in data.items()]
        headers = ["Metric", "Value"]
        return tabulate(table, headers, tablefmt="grid")
    else:
        return "Problems with data."