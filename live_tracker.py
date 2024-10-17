import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from functools import wraps
from tabulate import tabulate
from dotenv import load_dotenv
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ccxt.NetworkError as e:
            logging.error(f"Network error in {func.__name__}: {str(e)}")
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error in {func.__name__}: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {str(e)}")
    return wrapper

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

def create_bybit_connection(account_id):
    credentials = load_account_credentials()
    if account_id not in credentials:
        raise ValueError(f"No credentials found for account ID: {account_id}")
    
    account_info = credentials[account_id]
    
    return ccxt.bybit({
        'apiKey': account_info['api_key'],
        'secret': account_info['api_secret'],
        'enableRateLimit': True,
    })

class RiskAnalyzerConfig:
    def __init__(self):
        self.var_confidence_level = 0.95
        self.es_confidence_level = 0.95
        self.risk_free_rate = 0.01
        self.target_return = 0
        self.calmar_years = 3
        self.historical_data_limit = 30
        self.cache_expiry = 3600  # Cache expiry in seconds

class CryptoRiskAnalyzer:
    def __init__(self, config=None, account_name=None, account_id=None):
        self.exchange = create_bybit_connection(account_id)
        self.config = config or RiskAnalyzerConfig()
        self.account_name = account_name
        self.portfolio_value = 0
        self.positions = []
        self.historical_data = {}
        self.data_cache = {}
        self.risk_metrics = {}
        self.total_pnl = 0
        self.total_equity = 0
        self.free_equity = 0
        self.asset_weights = {}
        self.long_exposure = 0
        self.short_exposure = 0
        self.net_exposure = 0

    # DATA
    @error_handler
    def fetch_account_data(self):
        balance = self.exchange.fetchBalance()
        usdt_data = next((coin for coin in balance['info']['result']['list'][0]['coin'] if coin['coin'] == 'USDT'), None)
        
        if usdt_data:
            self.total_equity = float(usdt_data['equity'])
            self.total_pnl = float(usdt_data['unrealisedPnl'])
            self.free_equity = float(usdt_data['availableToWithdraw'])
        else:
            logging.error("USDT data not found in balance information")
            self.total_equity = 0
            self.total_pnl = 0
            self.free_equity = 0

        self.positions = self.exchange.fetchPositions() # if unified account, it fetches only 20 positions
        logging.info(f"Portfolio Value: ${self.total_equity:.2f}")
        logging.info(f"Unrealized PnL: ${self.total_pnl:.2f}")
        logging.info(f"Free Capital: ${self.free_equity:.2f}")

    @error_handler
    def fetch_historical_data(self, symbol, timeframe='1d', limit=30):
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if cache_key in self.data_cache and (pd.Timestamp.now() - self.data_cache[cache_key]['timestamp']).total_seconds() < self.config.cache_expiry:
            return self.data_cache[cache_key]['data']

        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        self.historical_data[symbol] = df
        self.data_cache[cache_key] = {'data': df, 'timestamp': pd.Timestamp.now()}
        return df

    def calculate_exposures(self):
        self.long_exposure = sum((float(position['contracts'])*float(position['markPrice'])) for position in self.positions if position['side'] == 'long')
        self.short_exposure = sum((float(position['contracts'])*float(position['markPrice'])) for position in self.positions if position['side'] == 'short')
        self.net_exposure = self.long_exposure - self.short_exposure
        
        logging.info(f"Long Exposure: ${self.long_exposure:.2f}")
        logging.info(f"Short Exposure: ${self.short_exposure:.2f}")
        logging.info(f"Net Exposure: ${self.net_exposure:.2f}")

    def calculate_asset_weights(self):
        total_exposure = self.long_exposure + self.short_exposure
        self.asset_weights = {
            position['symbol']: (abs(float(position['contracts'])*float(position['markPrice'])) / total_exposure, position['side'])
            for position in self.positions
        }

    def calculate_leverage_ratio(self):
        total_exposure = self.long_exposure + self.short_exposure
        leverage_ratio = total_exposure / self.total_equity
        logging.info(f"Leverage Ratio: {leverage_ratio:.2f}")
        return leverage_ratio

    def calculate_liquidation_risk(self):
        for position in self.positions:
            current_price = float(position['markPrice'])
            liquidation_price = float(position['liquidationPrice'])
            risk_percentage = (current_price - liquidation_price) / current_price
            logging.info(f"Liquidation Risk for {position['symbol']}: {risk_percentage*100:.2f}%")
            yield {'symbol': position['symbol'], 'risk_percentage': risk_percentage}
        
    # REPORT
    def generate_risk_report(self):
        print("\n--- Crypto Risk Analysis Report ---")
        print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.fetch_account_data()
        self.calculate_exposures()
        self.calculate_asset_weights()
        
        self.risk_metrics['Leverage'] = self.calculate_leverage_ratio()

        self.print_position_details()

    def print_account_summary(self):
        print("\nAccount Summary:")
        summary_table = [
            ["Unrealized PnL", f"${self.total_pnl:.2f}"],
            ["Total Equity", f"${self.total_equity:.2f}"],
            ["Free Capital", f"${self.free_equity:.2f}"],
            ["N. Positions", len(self.positions)],
            ["Leverage Ratio", f"{self.risk_metrics['Leverage']:.2f}"],
            ["Long Exposure", f"${self.long_exposure:.2f}"],
            ["Short Exposure", f"${self.short_exposure:.2f}"],
            ["Net Exp/Capital", f"{(self.net_exposure / self.total_equity):.2f}"]
        ]
        print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="grid"))

    def print_position_details(self):
        print(f"\nPosition Details ({len(self.positions)}):")
        position_data = []
        self.calculate_asset_weights()
        for position in self.positions:
            symbol = position['symbol']
            side = position['side']
            weight, _ = self.asset_weights[symbol]
            exposure = float(position['contracts'])*float(position['markPrice'])
            size = float(position['contracts'])
            entry_price = float(position['entryPrice'])
            mark_price = float(position['markPrice'])
            unrealized_pnl = float(position['unrealizedPnl'])
            liquidation_price = float(position['liquidationPrice'])
            liquidation_risk = (abs(mark_price - liquidation_price) / mark_price) * 100

            pnl_color = '\033[92m' if unrealized_pnl >= 0 else '\033[91m'
            pnl_formatted = f"{pnl_color}${unrealized_pnl:.2f}\033[0m"

            position_data.append([
                symbol, side, f"{exposure:.2f}", f"{size:.4f}", f"{weight*100:.2f}%", f"{entry_price:.4f}", f"{mark_price:.4f}",
                pnl_formatted, f"{liquidation_risk:.2f}%"
            ])

        headers = ["Symbol", "Side", "Exposure", "Size", "Weight", "Entry Price", "Mark Price", "UnPnL", "Liquidation Risk"]
        print(tabulate(position_data, headers=headers, tablefmt="grid"))
        
        tot_pnl_color = '\033[92m' if self.total_pnl >= 0 else '\033[91m'
        print(f"\nTotal PnL: {tot_pnl_color}${self.total_pnl:.2f}\033[0m")
        
        print("\nNote: Weight represents the proportion of the total portfolio exposure.")
        print("Note: Liquidation Risk represents the movement need in the opposite direction to reach the liquidation price.")

        self.print_account_summary()

def main():
    config = RiskAnalyzerConfig()
    
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
    
    selected_account_name = account_credentials[account_id].get('name', 'Unknown')
    analyzer = CryptoRiskAnalyzer(config, selected_account_name, account_id)
    analyzer.generate_risk_report()

if __name__ == "__main__":
    main()