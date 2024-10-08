import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from functools import wraps
from tabulate import tabulate
import sqlite3
from dotenv import dotenv_values, load_dotenv
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

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def fetch_account_data(self, account_name):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM daily_reports WHERE account_name = ?", (account_name,))
            return cursor.fetchone()

    def fetch_historical_equity(self, account_name):
        with self.get_connection() as conn:
            query = "SELECT date, equity FROM daily_reports WHERE account_name = ? ORDER BY date"
            return pd.read_sql_query(query, conn, params=(account_name,), index_col='date', parse_dates=['date'])
    
    def get_date_count(self, account_name):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT date) FROM daily_reports WHERE account_name = ?", (account_name,))
            return cursor.fetchone()[0]
        
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
    def __init__(self, config=None, db_manager=None, account_name=None, account_id=None):
        self.exchange = create_bybit_connection(account_id)
        self.config = config or RiskAnalyzerConfig()
        self.db_manager = db_manager
        self.days_in_db = 0
        self.account_name = account_name
        self.portfolio_value = 0
        self.positions = []
        self.historical_data = {}
        self.data_cache = {}
        self.risk_metrics = {}
        self.total_pnl = 0
        self.total_equity = 0
        self.asset_weights = {}
        self.historical_equity = None

    # DATA
    @error_handler
    def fetch_account_data(self):
        balance = self.exchange.fetchBalance()
        usdt_data = next((coin for coin in balance['info']['result']['list'][0]['coin'] if coin['coin'] == 'USDT'), None)
        
        if usdt_data:
            self.total_equity = float(usdt_data['equity'])
            self.total_pnl = float(usdt_data['unrealisedPnl'])
        else:
            logging.error("USDT data not found in balance information")
            self.total_equity = 0
            self.total_pnl = 0

        self.positions = self.exchange.fetchPositions()
        logging.info(f"Portfolio Value: ${self.total_equity:.2f}")
        logging.info(f"Unrealized PnL: ${self.total_pnl:.2f}")
        # logging.info(f"Portfolio Positions: {self.positions}")

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

    def align_historical_data(self):
        all_dates = pd.DatetimeIndex([])
        for df in self.historical_data.values():
            all_dates = all_dates.union(df.index)
        all_dates = all_dates.sort_values()

        aligned_data = {}
        for symbol, df in self.historical_data.items():
            aligned_df = df.reindex(all_dates)
            aligned_df['close'] = aligned_df['close'].ffill()  # Forward fill missing values
            aligned_data[symbol] = aligned_df

        self.historical_data = aligned_data

    def count_days_in_db(self):
        if self.db_manager:
            self.days_in_db = self.db_manager.get_date_count(self.account_name)
        else:
            self.days_in_db = 0

    # CALCULATIONS
    def set_historical_equity(self):
        if self.db_manager and self.account_name:
            self.historical_equity = self.db_manager.fetch_historical_equity(self.account_name)
            logging.info(f"Historical equity data fetched from database for account ID: {self.account_name}")
        else:
            logging.warning("Database manager or account ID not provided. Unable to fetch historical equity data.")

    def calculate_asset_weights(self):
        total_exposure = sum(abs(float(position['notional'])) for position in self.positions)
        self.asset_weights = {
            position['symbol']: (abs(float(position['notional'])) / total_exposure, position['side'])
            for position in self.positions
        }

    def calculate_weighted_portfolio_returns(self):
        self.align_historical_data()
        all_dates = sorted(set.union(*[set(df.index) for df in self.historical_data.values()]))
        weighted_returns = pd.Series(0, index=pd.DatetimeIndex(all_dates[1:]))
        
        for symbol, (weight, side) in self.asset_weights.items():
            prices = self.historical_data[symbol]['close'].dropna()
            returns = prices.pct_change().dropna()
            
            # Invert returns for short positions
            if side == 'short':
                returns = -returns
            
            weighted_returns += weight * returns.reindex(weighted_returns.index).fillna(0)
        
        return weighted_returns

    def calculate_portfolio_returns(self):
        if self.historical_equity is None or self.historical_equity.empty:
            logging.warning("No historical equity data available. Using weighted returns instead.")
            return self.calculate_weighted_portfolio_returns()
        
        returns = self.historical_equity.pct_change().dropna()
        return returns
    
    def calculate_var(self, confidence_level=0.95):
        returns = self.calculate_portfolio_returns()
        if returns.empty:
            logging.warning("No valid returns data for VaR calculation")
            return 0
        var = returns.quantile(1 - confidence_level)
        var_usd = abs(var * self.total_equity)
        var_usd_value = var_usd.item() if isinstance(var_usd, pd.Series) else var_usd
        logging.info(f"Value at Risk (VaR) at {confidence_level*100}% confidence: ${var_usd_value:.2f}")
        return var_usd_value

    def calculate_expected_shortfall(self, confidence_level=0.95):
        returns = self.calculate_portfolio_returns()
        if returns.empty:
            logging.warning("No valid returns data for ES calculation")
            return 0
        var = returns.quantile(1 - confidence_level)
        es = returns[returns <= var].mean()
        es_usd = abs(es * self.total_equity)
        es_usd_value = es_usd.item() if isinstance(es_usd, pd.Series) else es_usd
        logging.info(f"Weighted Expected Shortfall at {confidence_level*100}% confidence: ${es_usd_value:.2f}")
        return es_usd_value

    def calculate_sharpe_ratio(self, risk_free_rate=0.1, trading_days=365):
        returns = self.calculate_portfolio_returns()
        if returns.empty:
            logging.warning("No valid returns data for Sharpe ratio calculation")
            return 0
        excess_returns = returns - risk_free_rate / trading_days
        sharpe = (excess_returns.mean() * trading_days) / (returns.std() * np.sqrt(trading_days))
        sharpe_value = sharpe.item() if isinstance(sharpe, pd.Series) else sharpe
        logging.info(f"Sharpe Ratio: {sharpe_value:.2f}")
        return sharpe_value

    def calculate_sortino_ratio(self, risk_free_rate=0.1, target_return=0, trading_days=365):
        returns = self.calculate_portfolio_returns()
        if returns.empty:
            logging.warning("No valid returns data for Sortino ratio calculation")
            return 0
        downside_returns = returns[returns < target_return]
        excess_return = returns.mean() * trading_days - risk_free_rate
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(trading_days)
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        sortino_value = sortino_ratio.item() if isinstance(sortino_ratio, pd.Series) else sortino_ratio
        logging.info(f"Sortino Ratio: {sortino_value:.2f}")
        return sortino_value

    def calculate_omega_ratio(self, threshold=0):
        returns = self.calculate_portfolio_returns()
        if returns.empty:
            logging.warning("No valid returns data for Omega ratio calculation")
            return 0
        return_threshold = returns - threshold
        positive_returns = return_threshold[return_threshold > 0].sum()
        negative_returns = abs(return_threshold[return_threshold < 0].sum())
        
        # Convert to scalar values
        positive_returns = positive_returns.item() if isinstance(positive_returns, pd.Series) else positive_returns
        negative_returns = negative_returns.item() if isinstance(negative_returns, pd.Series) else negative_returns
        
        if negative_returns != 0:
            omega_ratio = positive_returns / negative_returns
        else:
            omega_ratio = float('inf')
        
        logging.info(f"Omega Ratio: {omega_ratio:.2f}")
        return omega_ratio

    def calculate_treynor_ratio(self, market_returns, risk_free_rate=0.01):
        portfolio_returns = self.calculate_portfolio_returns()
        beta = self.calculate_beta(market_returns)
        excess_return = portfolio_returns.mean() * 252 - risk_free_rate
        treynor_ratio = excess_return / beta if beta != 0 else 0
        treynor_value = treynor_ratio.item() if isinstance(treynor_ratio, pd.Series) else treynor_ratio
        logging.info(f"Treynor Ratio: {treynor_value:.2f}")
        return treynor_value

    def calculate_beta(self, market_returns):
        portfolio_returns = self.calculate_portfolio_returns()
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance if market_variance != 0 else 0
        beta_value = beta.item() if isinstance(beta, pd.Series) else beta
        logging.info(f"Portfolio Beta: {beta_value:.2f}")
        return beta_value

    def calculate_max_drawdown(self):
        if self.historical_equity is None or self.historical_equity.empty:
            logging.warning("No historical equity data available for max drawdown calculation")
            return 0
        
        peak = self.historical_equity.expanding(min_periods=1).max()
        drawdown = (self.historical_equity - peak) / peak
        max_drawdown = drawdown.min()
        max_drawdown_value = max_drawdown.item() if isinstance(max_drawdown, pd.Series) else max_drawdown
        logging.info(f"Maximum Drawdown: {max_drawdown_value*100:.2f}%")
        return max_drawdown_value

    def calculate_calmar_ratio(self, years=3):
        returns = self.calculate_portfolio_returns()
        if returns.empty:
            logging.warning("No valid returns data for Calmar ratio calculation")
            return 0
        total_return = (returns + 1).prod() - 1
        annualized_return = (1 + total_return) ** (365 / len(returns)) - 1
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        calmar_value = calmar_ratio.item() if isinstance(calmar_ratio, pd.Series) else calmar_ratio
        logging.info(f"Calmar Ratio: {calmar_value:.2f}")
        return calmar_value

    def calculate_leverage_ratio(self):
        total_position_value = sum(abs(float(position['notional'])) for position in self.positions)
        leverage_ratio = total_position_value / self.total_equity
        logging.info(f"Leverage Ratio: {leverage_ratio:.2f}")
        return leverage_ratio

    def calculate_liquidation_risk(self):
        for position in self.positions:
            current_price = float(position['markPrice'])
            liquidation_price = float(position['liquidationPrice'])
            risk_percentage = (current_price - liquidation_price) / current_price
            logging.info(f"Liquidation Risk for {position['symbol']}: {risk_percentage*100:.2f}%")
            yield {'symbol': position['symbol'], 'risk_percentage': risk_percentage}

    # PLOTS
    def plot_portfolio_performance(self):
        if self.historical_equity is None or self.historical_equity.empty:
            logging.warning("No historical equity data available for plotting portfolio performance")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.historical_equity.index, self.historical_equity.values)
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.tight_layout()
        
        # Create 'plots' folder if it doesn't exist
        plots_folder = 'plots'
        os.makedirs(plots_folder, exist_ok=True)
        
        # Save the figure in the 'plots' folder
        plt.savefig(os.path.join(plots_folder, 'portfolio_performance.png'))
        plt.close()

    def plot_drawdown(self):
        if self.historical_equity is None or self.historical_equity.empty:
            logging.warning("No historical equity data available for plotting drawdown")
            return

        peak = self.historical_equity.expanding(min_periods=1).max()
        drawdown = (self.historical_equity / peak) - 1
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown.values)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.tight_layout()
        
        # Create 'plots' folder if it doesn't exist
        plots_folder = 'plots'
        os.makedirs(plots_folder, exist_ok=True)
        
        # Save the figure in the 'plots' folder
        plt.savefig(os.path.join(plots_folder, 'portfolio_drawdown.png'))
        plt.close()

    # REPORT
    def generate_risk_report(self):
        print("\n--- Crypto Risk Analysis Report ---")
        print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.fetch_account_data()
        self.set_historical_equity()
        self.count_days_in_db()

        if self.historical_equity is None or self.historical_equity.empty:
            logging.warning("Historical equity data not available. Some calculations may be less accurate.")
            for position in self.positions:
                self.fetch_historical_data(position['symbol'])

        self.calculate_asset_weights()
        
        self.risk_metrics['VaR'] = self.calculate_var()
        self.risk_metrics['ES'] = self.calculate_expected_shortfall()
        self.risk_metrics['Sharpe'] = self.calculate_sharpe_ratio()
        self.risk_metrics['Sortino'] = self.calculate_sortino_ratio()
        self.risk_metrics['Omega'] = self.calculate_omega_ratio()
        self.risk_metrics['Max Drawdown'] = self.calculate_max_drawdown()
        self.risk_metrics['Calmar'] = self.calculate_calmar_ratio()
        self.risk_metrics['Leverage'] = self.calculate_leverage_ratio()

        self.print_position_details()
        self.print_risk_metrics()
        
        self.plot_portfolio_performance()
        self.plot_drawdown()

    def print_risk_metrics(self):
        print("\nRisk Metrics:")
        metrics_table = [
            ["Value at Risk (VaR)", f"${self.risk_metrics['VaR']:.2f}"],
            ["Expected Shortfall (ES)", f"${self.risk_metrics['ES']:.2f}"],
            ["Sharpe Ratio", f"{self.risk_metrics['Sharpe']:.2f}"],
            ["Sortino Ratio", f"{self.risk_metrics['Sortino']:.2f}"],
            ["Omega Ratio", f"{self.risk_metrics['Omega']:.2f}"],
            ["Maximum Drawdown", f"{self.risk_metrics['Max Drawdown']*100:.2f}%"],
            ["Calmar Ratio", f"{self.risk_metrics['Calmar']:.2f}"],
            ["Leverage Ratio", f"{self.risk_metrics['Leverage']:.2f}"]
        ]
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))
        print(f"\nNote: Days in the database: {self.days_in_db}\n")

    def print_account_summary(self):
        print("\nAccount Summary:")
        summary_table = [
            ["Unrealized PnL", f"${self.total_pnl:.2f}"],
            ["Total Equity", f"${self.total_equity:.2f}"],
            ["N. Positions", len(self.positions)]
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
            exposure = float(position['notional'])
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
    db_manager = DatabaseManager('database.db')
    
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
    analyzer = CryptoRiskAnalyzer(config, db_manager, selected_account_name, account_id)
    analyzer.generate_risk_report()

if __name__ == "__main__":
    main()