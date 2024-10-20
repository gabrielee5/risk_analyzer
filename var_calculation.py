import numpy as np
from pybit.unified_trading import HTTP
import pandas as pd
from datetime import datetime, timedelta
from dotenv import dotenv_values
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate
from general_functions import establish_session, get_account_keys, create_table, get_account_type

# to be added in the main file and to the report generator

def fetch_current_equity(session):
    account_type = get_account_type(session)
    response = session.get_wallet_balance(
    accountType=account_type)
    if account_type == "UNIFIED":
        equity = float(response['result']['list'][0]['totalEquity'])
    else:
        equity = float(response['result']['list'][0]['coin'][0]['equity'])
    
    return equity
    
def fetch_open_positions(session):
    try:
        # Fetch all open positions
        response = session.get_positions(
            category="linear",
            settleCoin="USDT"
        )
        
        # Check if the API call was successful
        if response['retCode'] != 0:
            raise Exception(f"API Error: {response['retMsg']}")
        
        # Extract the list of positions
        positions = response['result']['list']
        
        # Filter out positions with zero size
        active_positions = [
            position for position in positions 
            if float(position['size']) != 0
        ]
        # print(active_positions)

        # Convert to a more readable format
        formatted_positions = []
        for position in active_positions:
            formatted_positions.append({
                'symbol': position['symbol'],
                'side': "Long" if position['side'] == "Buy" else "Short",
                'avg_price': float(position['avgPrice']),
                'liq_price': float(position['liqPrice']),
                'leverage': float(position['leverage']),
                'exposure': (float(position['size']) * float(position['markPrice'])),
            })
        
        return formatted_positions
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
def calculate_portfolio_weights(positions):
    """
    Calculate the weight of each position in the portfolio based on its exposure.
    
    :param positions: List of dictionaries containing position information
    :return: Dictionary with symbols as keys and their corresponding weights as values
    """
    total_exposure = sum(abs(float(position['exposure'])) for position in positions)
    
    weights = {}
    for position in positions:
        symbol = position['symbol']
        exposure = abs(float(position['exposure']))
        weight = exposure / total_exposure
        # weight = exposure / equity
        weights[symbol] = weight
    
    return weights

def calculate_portfolio_leverage(equity, positions):
    total_exposure = sum(abs(float(position['exposure'])) for position in positions)
    leverage = total_exposure / equity
    return leverage

def fetch_historical_data(session, symbols, days=30, timef="D", cache_dir='data_cache'):
    try:
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        historical_data = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        for symbol in symbols:
            cache_file = os.path.join(cache_dir, f"{symbol}_{timef}.json")
            
            # Check if cached data exists and is recent
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                cached_end_time = cached_data['end_time']
                if end_time - cached_end_time < 24 * 60 * 60 * 1000:  # Less than 24 hours old
                    df = pd.DataFrame(cached_data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                        df[col] = df[col].astype(float)
                    historical_data[symbol] = df
                    print(f"Using cached data for {symbol}")
                    continue
            
            # Fetch new data from API
            response = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=timef,
                start=start_time,
                end=end_time
            )
            
            if response['retCode'] != 0:
                raise Exception(f"API Error for {symbol}: {response['retMsg']}")
            
            kline_data = response['result']['list']
            df = pd.DataFrame(kline_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)
            df = df.sort_values('timestamp')
            
            historical_data[symbol] = df
            
            # Cache the new data
            cache_data = {
                'end_time': end_time,
                'data': df.to_dict(orient='records')
            }
            # Convert timestamps to strings for JSON serialization
            for record in cache_data['data']:
                record['timestamp'] = record['timestamp'].isoformat()
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            print(f"Fetched and cached new data for {symbol}")
        
        return historical_data
    
    except Exception as e:
        print(f"An error occurred while fetching historical data: {str(e)}")
        return None
    
def calculate_returns(historical_data, positions):
    """
    Calculate daily returns for each symbol based on closing prices,
    accounting for long and short positions.
    
    :param historical_data: Dictionary with symbols as keys and DataFrames of historical data as values
    :param positions: List of dictionaries containing position information
    :return: DataFrame of daily returns for all symbols
    """
    all_returns = {}
    
    # Create a dictionary to store the position side (long or short) for each symbol
    position_sides = {position['symbol']: position['side'] for position in positions}
    
    for symbol, data in historical_data.items():
        # Calculate daily returns
        daily_returns = data['close'].pct_change()
        
        # Adjust returns based on position side
        if position_sides[symbol] == "Short":
            daily_returns = -1 * daily_returns
        
        # Drop the first row (NaN) and reset the index
        daily_returns = daily_returns.dropna().reset_index(drop=True)
        
        all_returns[symbol] = daily_returns
    
    # Create a DataFrame with all returns
    returns_df = pd.DataFrame(all_returns)
    
    return returns_df

def print_correlation_matrix(daily_returns):
    """
    Calculate and print a correlation matrix for the daily returns.
    Save the heatmap in the 'images' folder.
    
    :param daily_returns: DataFrame of daily returns for all symbols
    """
    correlation_matrix = daily_returns.corr()
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(2))  # Round to 2 decimal places for readability
    
    # Create 'images' folder if it doesn't exist
    images_folder = 'images'
    os.makedirs(images_folder, exist_ok=True)
    
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Daily Returns')
    plt.tight_layout()
    
    # Save the heatmap in the 'images' folder
    heatmap_path = os.path.join(images_folder, 'correlation_matrix.png')
    plt.savefig(heatmap_path)
    plt.close()  # Close the figure to free up memory
    
    print(f"\nCorrelation matrix heatmap has been saved as '{heatmap_path}'")

def calculate_portfolio_std_dev(daily_returns, portfolio_weights):
    """
    Calculate the portfolio standard deviation.
    
    :param daily_returns: DataFrame of daily returns for all symbols
    :param portfolio_weights: Dictionary with symbols as keys and their corresponding weights as values
    :return: Portfolio standard deviation
    """
    # Convert weights dictionary to a Series aligned with daily_returns columns
    weights = pd.Series(portfolio_weights).reindex(daily_returns.columns)
    
    # Calculate the covariance matrix
    cov_matrix = daily_returns.cov()
    
    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Calculate portfolio standard deviation
    portfolio_std_dev = np.sqrt(portfolio_variance)
    
    return portfolio_std_dev

def calculate_value_at_risk(leverage, portfolio_std_dev, confidence_level=0.95, time_horizon=1):
    """
    Calculate the Value at Risk (VaR) for a portfolio using the parametric method.
    
    :param leverage: Leverage of the portfolio, 1 if all the capital is invested and there is no lev
    :param portfolio_std_dev: standard deviation of the portfolio
    :param confidence_level: Confidence level for VaR calculation (default: 0.95 for 95% confidence)
    :param time_horizon: Time horizon for VaR in days (default: 1 day), if data is different then daily this needs mod
    :return: VaR value in currency units
    """
    # Calculate z-score for the given confidence level
    z_score = stats.norm.ppf(1 - confidence_level)
    
    # Calculate VaR
    var = z_score * portfolio_std_dev * np.sqrt(time_horizon) * leverage
    
    return abs(var)

def value_at_risk_data(session):
    confidence = 0.95
    timeframe = 24 # 24 if hourly data, 1 if daily
    candle_size = 60 # 60 hourly, D daily
    open_positions = fetch_open_positions(session)
    equity = fetch_current_equity(session)
    symbols = list(set([position['symbol'] for position in open_positions]))
    portfolio_weights = calculate_portfolio_weights(open_positions)
    historical_data = fetch_historical_data(session, symbols, 7, candle_size)
    leverage = calculate_portfolio_leverage(equity, open_positions)
    
    if historical_data:
        daily_returns = calculate_returns(historical_data, open_positions)
        # print_correlation_matrix(daily_returns)
        portfolio_std_dev = calculate_portfolio_std_dev(daily_returns, portfolio_weights) * np.sqrt(timeframe)
        annualized_volatility = portfolio_std_dev * np.sqrt(365)  # Assuming 365 trading days in a year, *24 if hourly data
        var = calculate_value_at_risk(leverage, portfolio_std_dev, confidence)
        
        return {
            "Confidence Level": f"{confidence*100}%",
            "Candles in day": f"{timeframe}",
            "Portfolio Leverage": f"{leverage:.2f}x",
            "Portfolio Std Dev (daily)": f"{portfolio_std_dev*100:.2f}%",
            "Annualized Volatility": f"{annualized_volatility*100:.2f}%",
            "Value at Risk": f"{var*100:.2f}%",
            "Monetary VaR": f"{var*equity:.2f}$"
        }
    else:
        return None
    
if __name__ == "__main__":
    key, secret = get_account_keys()
    session = establish_session(key, secret)
    data = value_at_risk_data(session)
    print(create_table(data))
