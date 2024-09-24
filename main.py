import ccxt
import pandas as pd
import numpy as np
from bybit_connection import create_bybit_connection
import functools
import matplotlib.pyplot as plt

def error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ccxt.NetworkError as e:
            print(f"Network error in {func.__name__}: {str(e)}")
        except ccxt.ExchangeError as e:
            print(f"Exchange error in {func.__name__}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error in {func.__name__}: {str(e)}")
    return wrapper

class RiskAnalyzerConfig:
    def __init__(self):
        self.var_confidence_level = 0.95
        self.es_confidence_level = 0.95
        self.risk_free_rate = 0.01
        self.target_return = 0
        self.calmar_years = 3
        self.historical_data_limit = 30

class CryptoRiskAnalyzer:
    def __init__(self, exchange, config=None):
        self.exchange = exchange
        self.config = config or RiskAnalyzerConfig()
        self.portfolio_value = 0
        self.positions = []
        self.historical_data = {}

    @error_handler
    def fetch_account_data(self):
        try:
            balance = self.exchange.fetch_balance()
            self.portfolio_value = balance['total']['USDT']
            self.positions = self.exchange.fetch_positions()
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Portfolio Positions: ${self.positions}")
        except Exception as e:
            print(f"Error fetching account data: {str(e)}")

    @error_handler
    def fetch_historical_data(self, symbol, timeframe='1d', limit=30):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            self.historical_data[symbol] = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.historical_data[symbol]['timestamp'] = pd.to_datetime(self.historical_data[symbol]['timestamp'], unit='ms')
            self.historical_data[symbol].set_index('timestamp', inplace=True)
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {str(e)}")

    @error_handler
    def calculate_var(self, confidence_level=0.95):
        returns = self.calculate_portfolio_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        var_usd = abs(var * self.portfolio_value)
        print(f"Value at Risk (VaR) at {confidence_level*100}% confidence: ${var_usd:.2f}")
        return var_usd

    @error_handler
    def calculate_expected_shortfall(self, confidence_level=0.95):
        returns = self.calculate_portfolio_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        es = returns[returns <= var].mean()
        es_usd = abs(es * self.portfolio_value)
        print(f"Expected Shortfall at {confidence_level*100}% confidence: ${es_usd:.2f}")
        return es_usd

    @error_handler
    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        returns = self.calculate_portfolio_returns()
        excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days
        sharpe = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
        print(f"Sharpe Ratio: {sharpe:.2f}")
        return sharpe
    
    @error_handler
    def calculate_sortino_ratio(self, risk_free_rate=0.01, target_return=0):
        returns = self.calculate_portfolio_returns()
        downside_returns = returns[returns < target_return]
        excess_return = returns.mean() * 252 - risk_free_rate
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        sortino_ratio = excess_return / downside_deviation
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        return sortino_ratio

    @error_handler
    def calculate_beta(self, market_returns):
        portfolio_returns = self.calculate_portfolio_returns()
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance
        print(f"Portfolio Beta: {beta:.2f}")
        return beta

    @error_handler
    def calculate_calmar_ratio(self, years=3):
        returns = self.calculate_portfolio_returns()
        total_return = (returns + 1).prod() - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = annualized_return / abs(max_drawdown)
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        return calmar_ratio

    @error_handler
    def calculate_max_drawdown(self):
        cumulative_returns = (1 + self.calculate_portfolio_returns()).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
        return max_drawdown

    @error_handler
    def calculate_portfolio_returns(self):
        symbol = list(self.historical_data.keys())[0]
        prices = self.historical_data[symbol]['close'].values
        returns = np.diff(prices) / prices[:-1]
        return pd.Series(returns, index=self.historical_data[symbol].index[1:])

    @error_handler
    def calculate_leverage_ratio(self):
        total_position_value = sum(abs(float(position['notional'])) for position in self.positions)
        leverage_ratio = total_position_value / self.portfolio_value
        print(f"Leverage Ratio: {leverage_ratio:.2f}")
        return leverage_ratio

    @error_handler
    def calculate_liquidation_risk(self):
        for position in self.positions:
            entry_price = float(position['entryPrice'])
            current_price = float(position['markPrice'])
            leverage = float(position['leverage'])
            liquidation_price = entry_price * (1 - 1/leverage)
            risk_percentage = (current_price - liquidation_price) / current_price
            print(f"Liquidation Risk for {position['symbol']}: {risk_percentage*100:.2f}%")
            yield {'symbol': position['symbol'], 'risk_percentage': risk_percentage}

    @error_handler
    def plot_portfolio_performance(self):
        returns = self.calculate_portfolio_returns()
        cumulative_returns = (1 + returns).cumprod()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title('Portfolio Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.show()

    @error_handler
    def plot_drawdown(self):
        returns = self.calculate_portfolio_returns()
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        plt.figure(figsize=(10, 6))
        plt.plot(drawdown.index, drawdown.values)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.show()

    @error_handler
    def generate_risk_report(self):
        print("\n--- Crypto Risk Analysis Report ---\n")
        self.fetch_account_data()
        for position in self.positions:
            self.fetch_historical_data(position['symbol'])
        
        self.calculate_var()
        self.calculate_expected_shortfall()
        self.calculate_sharpe_ratio()
        self.calculate_max_drawdown()
        self.calculate_leverage_ratio()
        list(self.calculate_liquidation_risk())  # Consume the generator

def main():
    bybit = create_bybit_connection()
    config = RiskAnalyzerConfig()
    config.var_confidence_level = 0.99  # Example of changing a config parameter
    analyzer = CryptoRiskAnalyzer(bybit, config)
    analyzer.generate_risk_report()
    # analyzer.plot_portfolio_performance()
    # analyzer.plot_drawdown()

if __name__ == "__main__":
    main()