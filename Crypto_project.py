import requests
import pandas as pd
import ta
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
COINS = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD', 'ADA-USD', 'SOL-USD']
SMA_SHORT = 10
SMA_LONG = 50
RSI_PERIOD = 14
UPDATE_INTERVAL = 60  # seconds between price fetches
TRADE_LOG_FILE = "trade_log.csv"
TRADE_FRACTION = 0.1
RETRAIN_INTERVAL = 10  # retrain ML models every N new price points

# -----------------------------
# TEST PORTFOLIO
# -----------------------------
cash = 10000.0
portfolio = {coin: 0.0 for coin in COINS}

# Create trade log if doesn't exist
if not os.path.exists(TRADE_LOG_FILE):
    pd.DataFrame(columns=['timestamp', 'coin', 'action', 'qty', 'price', 'portfolio_value']).to_csv(TRADE_LOG_FILE, index=False)

# -----------------------------
# FUNCTIONS
# -----------------------------
def fetch_live_price(coin):
    """
    Fetches the latest price for a cryptocurrency from Coinbase API.
    
    Args:
        coin (str): Cryptocurrency symbol, e.g., 'BTC-USD'
        
    Returns:
        float: Latest price
    """
    try:
        url = f"https://api.exchange.coinbase.com/products/{coin}/ticker"
        resp = requests.get(url)
        resp.raise_for_status()
        return float(resp.json()['price'])
    except Exception as e:
        print(f"Error fetching {coin} price: {e}")
        return None

def fetch_historical_data(coin):
    """
    Fetches historical daily OHLCV data for a cryptocurrency from Coinbase API.
    
    Args:
        coin (str): Cryptocurrency symbol
        
    Returns:
        pd.DataFrame: DataFrame with timestamp and closing price
    """
    url = f"https://api.exchange.coinbase.com/products/{coin}/candles"
    params = {'granularity': 86400}  # daily
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json(), columns=['time','low','high','open','close','volume'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={'close': f'{coin}_price'}, inplace=True)
        return df[['timestamp', f'{coin}_price']]
    except Exception as e:
        print(f"Error fetching historical data for {coin}: {e}")
        return pd.DataFrame(columns=['timestamp', f'{coin}_price'])

def compute_indicators(df, coin):
    """
    Computes technical indicators and signals for a given coin.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        coin (str): Cryptocurrency symbol
        
    Returns:
        pd.DataFrame: Updated DataFrame with indicators
    """
    price_col = f'{coin}_price'
    df[f'{coin}_SMA_SHORT'] = ta.trend.sma_indicator(df[price_col], SMA_SHORT)
    df[f'{coin}_SMA_LONG'] = ta.trend.sma_indicator(df[price_col], SMA_LONG)
    df[f'{coin}_RSI'] = ta.momentum.rsi(df[price_col], RSI_PERIOD)
    df[f'{coin}_MACD'] = ta.trend.macd(df[price_col])
    df[f'{coin}_MACD_signal'] = ta.trend.macd_signal(df[price_col])

    # Signals
    df[f'{coin}_SMA_signal'] = np.where(df[f'{coin}_SMA_SHORT']>df[f'{coin}_SMA_LONG'],1,
                                       np.where(df[f'{coin}_SMA_SHORT']<df[f'{coin}_SMA_LONG'],-1,0))
    df[f'{coin}_RSI_signal'] = np.where(df[f'{coin}_RSI']<30,1,np.where(df[f'{coin}_RSI']>70,-1,0))
    df[f'{coin}_MACD_signal_flag'] = np.where(df[f'{coin}_MACD']>df[f'{coin}_MACD_signal'],1,
                                             np.where(df[f'{coin}_MACD']<df[f'{coin}_MACD_signal'],-1,0))
    df.ffill(inplace=True)
    return df

def train_models(df, coin):
    """
    Trains Random Forest, XGBoost, and SVM models and computes validation accuracy.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target
        coin (str): Cryptocurrency symbol
        
    Returns:
        dict: Dictionary {model_name: (model, val_accuracy)}
    """
    features = [f'{coin}_SMA_signal', f'{coin}_RSI_signal', f'{coin}_MACD_signal_flag']
    df = df.dropna()
    if len(df) < 10:
        return {}

    X = df[features].iloc[:-1]
    y = (df[f'{coin}_price'].shift(-1) > df[f'{coin}_price']).astype(int).iloc[:-1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define models
    rf = RandomForestClassifier(n_estimators=100, max_depth=None)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    svm_model = SVC(probability=True)

    models = {'RandomForest': rf, 'XGBoost': xgb, 'SVM': svm_model}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        results[name] = (model, acc)

    return results

def ensemble_predict(models_dict, latest_features):
    """
    Produces a weighted ensemble prediction based on validation accuracy.
    
    Args:
        models_dict (dict): Dictionary of trained models and accuracies
        latest_features (pd.DataFrame): Single-row DataFrame of latest features
        
    Returns:
        int: 1=BUY, 0=SELL/HOLD
    """
    total_weight = sum(acc for _, acc in models_dict.values())
    if total_weight == 0:
        return 1  # default BUY if no models

    weighted_vote = 0
    for model, acc in models_dict.values():
        pred = model.predict(latest_features)[0]
        weighted_vote += pred * acc

    weighted_vote /= total_weight
    return 1 if weighted_vote > 0.5 else 0

def log_trade(timestamp, coin, action, qty, price, portfolio_value):
    """
    Appends trade details to CSV log.
    """
    pd.DataFrame([{
        'timestamp': timestamp, 'coin': coin, 'action': action, 'qty': qty,
        'price': price, 'portfolio_value': portfolio_value
    }]).to_csv(TRADE_LOG_FILE, mode='a', index=False, header=False)

def print_portfolio_summary(cash, portfolio, all_data):
    """
    Prints cash, coin holdings, and total portfolio value.
    """
    print("\n=== Portfolio Summary ===")
    print(f"Cash: ${cash:.2f}")
    total_value = cash
    for coin in COINS:
        price = all_data[coin].iloc[-1][f"{coin}_price"] if not all_data[coin].empty else 0
        value = portfolio[coin] * price
        total_value += value
        print(f"{coin}: {portfolio[coin]:.6f} | Price=${price:.2f} | Value=${value:.2f}")
    print(f"Total Portfolio Value: ${total_value:.2f}")
    print("========================\n")

# -----------------------------
# LOAD HISTORICAL DATA
# -----------------------------
all_data = {}
ml_models = {}
ml_counters = {}

for coin in COINS:
    df = fetch_historical_data(coin)
    if not df.empty:
        df = compute_indicators(df, coin)
    all_data[coin] = df
    ml_models[coin] = None
    ml_counters[coin] = 0

# -----------------------------
# LIVE LOOP
# -----------------------------
print("=== Starting Live Fractional Test Portfolio ===")
while True:
    timestamp = datetime.utcnow()
    portfolio_value = cash + sum(
        portfolio[c]*all_data[c].iloc[-1][f"{c}_price"] if not all_data[c].empty else 0
        for c in COINS
    )

    for coin in COINS:
        price = fetch_live_price(coin)
        if price is None:
            continue

        # Append latest price
        new_row = pd.DataFrame({'timestamp':[timestamp], f'{coin}_price':[price]})
        all_data[coin] = pd.concat([all_data[coin], new_row], ignore_index=True)
        all_data[coin] = compute_indicators(all_data[coin], coin)

        # Train ML models periodically
        ml_counters[coin] += 1
        if ml_counters[coin] >= RETRAIN_INTERVAL:
            ml_models[coin] = train_models(all_data[coin], coin)
            ml_counters[coin] = 0

        # Generate trade signal
        features = [f'{coin}_SMA_signal', f'{coin}_RSI_signal', f'{coin}_MACD_signal_flag']
        last_features = all_data[coin][features].iloc[[-1]]

        signal = ensemble_predict(ml_models[coin], last_features) if ml_models[coin] else 1

        # Execute fractional trade
        qty = TRADE_FRACTION * cash / price if signal == 1 else TRADE_FRACTION * portfolio[coin]
        if signal == 1 and cash >= qty*price:
            cash -= qty*price
            portfolio[coin] += qty
            log_trade(timestamp, coin, 'BUY', qty, price, portfolio_value)
        elif signal == 0 and portfolio[coin] >= qty:
            cash += qty*price
            portfolio[coin] -= qty
            log_trade(timestamp, coin, 'SELL', qty, price, portfolio_value)

    print_portfolio_summary(cash, portfolio, all_data)
    time.sleep(UPDATE_INTERVAL)
