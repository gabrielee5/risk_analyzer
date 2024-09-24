import ccxt
import pandas as pd
import numpy as np
from bybit_connection import create_bybit_connection

class CryptoRiskAnalyzer:
    def __init__(self, exchange):
        self.exchange = exchange
        self.portfolio_value = 0
        self.positions = []
        self.historical_data = {}

    def fetch_account_data(self):
        try:
            balance = self.exchange.fetch_balance()
            self.portfolio_value = balance['total']['USDT']
            self.positions = self.exchange.fetch_positions()
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        except Exception as e:
            print(f"Error fetching account data: {str(e)}")

    def fetch_historical_data(self, symbol, timeframe='1d', limit=30):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            self.historical_data[symbol] = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.historical_data[symbol]['timestamp'] = pd.to_datetime(self.historical_data[symbol]['timestamp'], unit='ms')
            self.historical_data[symbol].set_index('timestamp', inplace=True)
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {str(e)}")

    def calculate_var(self, confidence_level=0.95):
        returns = self.calculate_portfolio_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        var_usd = abs(var * self.portfolio_value)
        print(f"Value at Risk (VaR) at {confidence_level*100}% confidence: ${var_usd:.2f}")
        return var_usd

    def calculate_expected_shortfall(self, confidence_level=0.95):
        returns = self.calculate_portfolio_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        es = returns[returns <= var].mean()
        es_usd = abs(es * self.portfolio_value)
        print(f"Expected Shortfall at {confidence_level*100}% confidence: ${es_usd:.2f}")
        return es_usd

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        returns = self.calculate_portfolio_returns()
        excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days
        sharpe = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
        print(f"Sharpe Ratio: {sharpe:.2f}")
        return sharpe

    def calculate_max_drawdown(self):
        cumulative_returns = (1 + self.calculate_portfolio_returns()).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
        return max_drawdown

    def calculate_portfolio_returns(self):
        # This is a simplified version. In reality, you'd need to calculate
        # weighted returns based on your portfolio composition
        symbol = list(self.historical_data.keys())[0]  # Using the first symbol as an example
        return self.historical_data[symbol]['close'].pct_change().dropna()

    def calculate_leverage_ratio(self):
        total_position_value = sum(abs(float(position['notional'])) for position in self.positions)
        leverage_ratio = total_position_value / self.portfolio_value
        print(f"Leverage Ratio: {leverage_ratio:.2f}")
        return leverage_ratio

    def calculate_liquidation_risk(self):
        for position in self.positions:
            entry_price = float(position['entryPrice'])
            current_price = float(position['markPrice'])
            leverage = float(position['leverage'])
            liquidation_price = entry_price * (1 - 1/leverage)
            risk_percentage = (current_price - liquidation_price) / current_price
            print(f"Liquidation Risk for {position['symbol']}: {risk_percentage*100:.2f}%")
            yield {'symbol': position['symbol'], 'risk_percentage': risk_percentage}

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
    analyzer = CryptoRiskAnalyzer(bybit)
    analyzer.generate_risk_report()

if __name__ == "__main__":
    main()