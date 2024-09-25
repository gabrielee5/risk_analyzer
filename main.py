import ccxt
import pandas as pd
import numpy as np
from bybit_connection import create_bybit_connection
import matplotlib.pyplot as plt
import logging
import asyncio
import aiohttp
import os
from functools import wraps
from tabulate import tabulate

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
    def __init__(self, exchange, config=None):
        self.exchange = exchange
        self.config = config or RiskAnalyzerConfig()
        self.portfolio_value = 0
        self.positions = []
        self.historical_data = {}
        self.data_cache = {}
        self.risk_metrics = {}

    @error_handler
    def fetch_account_data(self):
        balance = self.exchange.fetch_balance()
        self.portfolio_value = balance['total']['USDT']
        self.positions = self.exchange.fetch_positions()
        logging.info(f"Portfolio Value: ${self.portfolio_value:.2f}")
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

    def calculate_portfolio_returns(self):
        symbol = list(self.historical_data.keys())[0]
        prices = self.historical_data[symbol]['close'].values
        returns = np.diff(prices) / prices[:-1]
        return pd.Series(returns, index=self.historical_data[symbol].index[1:])

    def calculate_var(self, confidence_level=0.95):
        returns = self.calculate_portfolio_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        var_usd = abs(var * self.portfolio_value)
        logging.info(f"Value at Risk (VaR) at {confidence_level*100}% confidence: ${var_usd:.2f}")
        return var_usd

    def calculate_expected_shortfall(self, confidence_level=0.95):
        returns = self.calculate_portfolio_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        es = returns[returns <= var].mean()
        es_usd = abs(es * self.portfolio_value)
        logging.info(f"Expected Shortfall at {confidence_level*100}% confidence: ${es_usd:.2f}")
        return es_usd

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        returns = self.calculate_portfolio_returns()
        excess_returns = returns - risk_free_rate / 252
        sharpe = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
        logging.info(f"Sharpe Ratio: {sharpe:.2f}")
        return sharpe

    def calculate_sortino_ratio(self, risk_free_rate=0.01, target_return=0):
        returns = self.calculate_portfolio_returns()
        downside_returns = returns[returns < target_return]
        excess_return = returns.mean() * 252 - risk_free_rate
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        sortino_ratio = excess_return / downside_deviation
        logging.info(f"Sortino Ratio: {sortino_ratio:.2f}")
        return sortino_ratio

    def calculate_omega_ratio(self, threshold=0):
        returns = self.calculate_portfolio_returns()
        return_threshold = returns - threshold
        positive_returns = return_threshold[return_threshold > 0].sum()
        negative_returns = abs(return_threshold[return_threshold < 0].sum())
        omega_ratio = positive_returns / negative_returns
        logging.info(f"Omega Ratio: {omega_ratio:.2f}")
        return omega_ratio

    def calculate_treynor_ratio(self, market_returns, risk_free_rate=0.01):
        portfolio_returns = self.calculate_portfolio_returns()
        beta = self.calculate_beta(market_returns)
        excess_return = portfolio_returns.mean() * 252 - risk_free_rate
        treynor_ratio = excess_return / beta
        logging.info(f"Treynor Ratio: {treynor_ratio:.2f}")
        return treynor_ratio

    def calculate_beta(self, market_returns):
        portfolio_returns = self.calculate_portfolio_returns()
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance
        logging.info(f"Portfolio Beta: {beta:.2f}")
        return beta

    def calculate_max_drawdown(self):
        cumulative_returns = (1 + self.calculate_portfolio_returns()).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        logging.info(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
        return max_drawdown

    def calculate_calmar_ratio(self, years=3):
        returns = self.calculate_portfolio_returns()
        total_return = (returns + 1).prod() - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = annualized_return / abs(max_drawdown)
        logging.info(f"Calmar Ratio: {calmar_ratio:.2f}")
        return calmar_ratio

    def calculate_leverage_ratio(self):
        total_position_value = sum(abs(float(position['notional'])) for position in self.positions)
        leverage_ratio = total_position_value / self.portfolio_value
        logging.info(f"Leverage Ratio: {leverage_ratio:.2f}")
        return leverage_ratio

    def calculate_liquidation_risk(self):
        for position in self.positions:
            entry_price = float(position['entryPrice'])
            current_price = float(position['markPrice'])
            leverage = float(position['leverage'])
            liquidation_price = entry_price * (1 - 1/leverage)
            risk_percentage = (current_price - liquidation_price) / current_price
            logging.info(f"Liquidation Risk for {position['symbol']}: {risk_percentage*100:.2f}%")
            yield {'symbol': position['symbol'], 'risk_percentage': risk_percentage}

    def plot_portfolio_performance(self):
        returns = self.calculate_portfolio_returns()
        cumulative_returns = (1 + returns).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title('Portfolio Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.tight_layout()
        
        # Create 'plots' folder if it doesn't exist
        plots_folder = 'plots'
        os.makedirs(plots_folder, exist_ok=True)
        
        # Save the figure in the 'plots' folder
        plt.savefig(os.path.join(plots_folder, 'portfolio_performance.png'))
        plt.close()

    def plot_drawdown(self):
        returns = self.calculate_portfolio_returns()
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
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

    def generate_risk_report(self):
        print("\n--- Crypto Risk Analysis Report ---")
        print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.fetch_account_data()
        for position in self.positions:
            self.fetch_historical_data(position['symbol'])
        
        self.risk_metrics['VaR'] = self.calculate_var()
        self.risk_metrics['ES'] = self.calculate_expected_shortfall()
        self.risk_metrics['Sharpe'] = self.calculate_sharpe_ratio()
        self.risk_metrics['Sortino'] = self.calculate_sortino_ratio()
        self.risk_metrics['Omega'] = self.calculate_omega_ratio()
        self.risk_metrics['Max Drawdown'] = self.calculate_max_drawdown()
        self.risk_metrics['Calmar'] = self.calculate_calmar_ratio()
        self.risk_metrics['Leverage'] = self.calculate_leverage_ratio()

        self.print_portfolio_summary()
        self.print_risk_metrics()
        self.print_position_details()
        
        self.plot_portfolio_performance()
        self.plot_drawdown()

    def print_portfolio_summary(self):
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Number of Positions: {len(self.positions)}\n")

    def print_risk_metrics(self):
        print("Risk Metrics:")
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
        print()

    def print_position_details(self):
        print("Position Details:")
        position_data = []
        tot_pnl = 0
        for position in self.positions:
            symbol = position['symbol']
            side = "Long" if position['side'] == 'long' else "Short"
            size = position['contracts']
            entry_price = position['entryPrice']
            mark_price = position['markPrice']
            unrealized_pnl = position['unrealizedPnl']
            cur_realized_pnl = position['info'].get('curRealisedPnl', 0)  # Use .get() with a default value
            pnl = float(unrealized_pnl) + float(cur_realized_pnl)  # Convert to float to ensure numeric addition
            liquidation_price = position['liquidationPrice']
            liquidation_risk = (abs(float(mark_price) - float(liquidation_price)) / float(mark_price)) * 100
            tot_pnl += pnl

            # Format PnL with color (green for positive, red for negative)
            pnl_color = '\033[92m' if pnl >= 0 else '\033[91m'  # Green if positive, Red if negative
            pnl_formatted = f"{pnl_color}${pnl:.2f}\033[0m"  # \033[0m resets the color

            position_data.append([
                symbol, side, size, f"{entry_price:.4f}", f"{mark_price:.4f}",
                pnl_formatted, f"{liquidation_risk:.2f}%"
            ])

        headers = ["Symbol", "Side", "Size", "Entry Price", "Mark Price", "PnL", "Liquidation Risk"]
        print(tabulate(position_data, headers=headers, tablefmt="grid"))
        
        # Print total PnL
        tot_pnl_color = '\033[92m' if tot_pnl >= 0 else '\033[91m'
        print(f"\nTotal PnL: {tot_pnl_color}${tot_pnl:.2f}\033[0m")
        
        print("\nNote: Liquidation Risk represents the percentage difference between the current price and the liquidation price.")
        print("Note: PnL includes both unrealized and current realized profit/loss.")


def main():
    bybit = create_bybit_connection()
    config = RiskAnalyzerConfig()
    analyzer = CryptoRiskAnalyzer(bybit, config)
    analyzer.generate_risk_report()

if __name__ == "__main__":
    main()