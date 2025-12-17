import requests
import pandas as pd
from datetime import datetime
import time

# Binance Public API Endpoints
BINANCE_BASE_URL = "https://api.binance.com/api/v3"

def get_binance_klines(symbol, interval='1d', limit=500):
    """
    Fetch historical k-line (candlestick) data from Binance.
    """
    try:
        # Binance expects pairs like BTCUSDT
        # Ensure symbol is correct format
        clean_symbol = symbol.upper().replace("-", "").replace("/", "")
        if not clean_symbol.endswith('USDT'):
             # If just BTC, append USDT
             clean_symbol += 'USDT'
             
        url = f"{BINANCE_BASE_URL}/klines"
        params = {
            'symbol': clean_symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Binance returns list of lists:
        # [Open time, Open, High, Low, Close, Volume, Close time, ...]
        df = pd.DataFrame(data, columns=[
            'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close_time', 'Quote_asset_val', 'Num_trades', 'Taker_buy_base', 'Taker_buy_quote', 'Ignore'
        ])
        
        # Convert timestamp to date
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        
        # Convert numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        print(f"Binance Klines Error: {e}")
        return None

def get_binance_price(symbol):
    """
    Fetch current price from Binance
    """
    try:
        clean_symbol = symbol.upper().replace("-", "").replace("/", "")
        if not clean_symbol.endswith('USDT'):
             clean_symbol += 'USDT'
             
        url = f"{BINANCE_BASE_URL}/ticker/price"
        params = {'symbol': clean_symbol}
        
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        data = response.json()
        
        return float(data['price'])
    except Exception as e:
        print(f"Binance Price Error: {e}")
        return None

def get_binance_24hr_stats(symbol):
    """
    Fetch 24hr ticker stats
    """
    try:
        clean_symbol = symbol.upper().replace("-", "").replace("/", "")
        if not clean_symbol.endswith('USDT'):
             clean_symbol += 'USDT'
             
        url = f"{BINANCE_BASE_URL}/ticker/24hr"
        params = {'symbol': clean_symbol}
        
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        data = response.json()
        
        return {
            'price': float(data['lastPrice']),
            'change_24h': float(data['priceChange']),
            'change_percent_24h': float(data['priceChangePercent']),
            'high_24h': float(data['highPrice']),
            'low_24h': float(data['lowPrice']),
            'volume_24h': float(data['volume']),
            'volume_quote_24h': float(data['quoteVolume'])
        }
    except Exception as e:
        print(f"Binance 24hr Stats Error: {e}")
        return None
