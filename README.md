# Crypto Trading Bot

This Python project simulates a live fractional cryptocurrency trading portfolio using technical indicators and machine learning models. It fetches live prices from the Coinbase API and executes trades in a simulated environment.

## Features

- **Supported Cryptocurrencies:** BTC, ETH, LTC, DOGE, ADA, SOL (adjustable)
- **Technical Indicators:**
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - MACD and MACD Signal
- **Machine Learning Models:**
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - Ensemble predictions weighted by validation accuracy
- **Fractional Trading:** Buy or sell a fraction of the portfolio based on signals
- **Logging:** All trades are recorded to `trade_log.csv` for analysis
- **Portfolio Summary:** Prints cash, coin holdings, and total portfolio value after each update

  
<img width="1024" height="1024" alt="Crypto_diagram" src="https://github.com/user-attachments/assets/822b26db-b2c6-4eac-aa61-94d5128ccc12" />


## Performance

<img width="1960" height="1154" alt="model_performance" src="https://github.com/user-attachments/assets/caf00ae6-6bda-4b7e-b61f-c34d3552669a" />


## Installation

1. Clone this repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
Install required packages:

bash
Copy
pip install pandas requests ta scikit-learn xgboost numpy
Usage
Update configuration variables in the script:

COINS: list of cryptocurrencies to trade

SMA_SHORT, SMA_LONG, RSI_PERIOD: technical indicator settings

UPDATE_INTERVAL: time between price fetches (in seconds)

TRADE_FRACTION: fraction of cash/holdings to trade

RETRAIN_INTERVAL: how often to retrain ML models

Run the bot:

bash
Copy
python crypto_trading_bot.py
View portfolio summaries and trades in the console. Trades will be saved in trade_log.csv.

File Structure
crypto_trading_bot.py – main trading script

trade_log.csv – generated log of trades

README.md – project documentation

Notes
This bot is simulation-only and does not execute real trades.

Make sure you have an active internet connection to fetch live price data from Coinbase API.

Adjust configuration parameters to experiment with different strategies.

License
This project is released under the MIT License.
