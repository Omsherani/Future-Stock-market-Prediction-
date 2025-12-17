import requests
from datetime import datetime, timedelta
import pandas as pd

# CoinGecko API endpoints (free tier, no API key needed)
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Mapping of common crypto symbols to CoinGecko IDs
CRYPTO_ID_MAP = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'ADA': 'cardano',
    'DOT': 'polkadot',
    'DOGE': 'dogecoin',
    'MATIC': 'matic-network',
    'LINK': 'chainlink',
    'UNI': 'uniswap',
    'AVAX': 'avalanche-2',
    'XRP': 'ripple',
    'LTC': 'litecoin',
    'BCH': 'bitcoin-cash',
    'ATOM': 'cosmos',
    'XLM': 'stellar',
    'ALGO': 'algorand',
    'VET': 'vechain',
    'FIL': 'filecoin',
    'TRX': 'tron',
    'TRX': 'tron',
    'ETC': 'ethereum-classic',
    'XAUUSD': 'tether-gold', # Map Gold to Tether Gold (XAUt)
    'GOLD': 'tether-gold',
    'PAXG': 'pax-gold',
    'XAUt': 'tether-gold'
}

def is_crypto_symbol(symbol):
    """Check if symbol is a known cryptocurrency."""
    base_symbol = symbol.replace('-USD', '').upper()
    return base_symbol in CRYPTO_ID_MAP

def get_coingecko_id(symbol):
    """Get CoinGecko ID from symbol."""
    base_symbol = symbol.replace('-USD', '').upper()
    return CRYPTO_ID_MAP.get(base_symbol)

def fetch_crypto_current_price(symbol):
    """Fetch current price and 24h stats from CoinGecko."""
    coin_id = get_coingecko_id(symbol)
    if not coin_id:
        return None
    
    try:
        url = f"{COINGECKO_BASE_URL}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true',
            'include_last_updated_at': 'true'
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if coin_id in data:
            return {
                'price': data[coin_id]['usd'],
                'change_24h': data[coin_id].get('usd_24h_change', 0),
                'volume_24h': data[coin_id].get('usd_24h_vol', 0),
                'last_updated': data[coin_id].get('last_updated_at', None)
            }
    except Exception as e:
        print(f"CoinGecko API error: {e}")
        return None

def fetch_crypto_historical_data(symbol, days=365):
    """Fetch historical price data from CoinGecko."""
    coin_id = get_coingecko_id(symbol)
    if not coin_id:
        return None
    
    try:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days
        }
        
        # Only add interval for longer periods
        if days > 90:
            params['interval'] = 'daily'
        
        print(f"Fetching from CoinGecko: {url} with params {params}")
        response = requests.get(url, params=params, timeout=10)
        
        # Check for rate limiting
        if response.status_code == 429:
            print("CoinGecko rate limit hit, will use fallback")
            return None
            
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame format similar to yfinance
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not prices:
            print("No prices in response")
            return None
        
        df_data = []
        for i, (timestamp, price) in enumerate(prices):
            date = datetime.fromtimestamp(timestamp / 1000)
            volume = volumes[i][1] if i < len(volumes) else 0
            
            df_data.append({
                'Date': date,
                'Open': price,  # CoinGecko doesn't provide OHLC for free tier, using close as approximation
                'High': price * 1.01,  # Approximate
                'Low': price * 0.99,   # Approximate
                'Close': price,
                'Volume': volume
            })
        
        df = pd.DataFrame(df_data)
        print(f"Successfully fetched {len(df)} data points from CoinGecko")
        return df
        
    except Exception as e:
        import traceback
        print(f"CoinGecko historical data error: {e}")
        print(f"Response status: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"Response text: {response.text[:200] if 'response' in locals() else 'N/A'}")
        traceback.print_exc()
        return None

def get_crypto_info(symbol):
    """Get cryptocurrency information."""
    coin_id = get_coingecko_id(symbol)
    if not coin_id:
        return None
    
    try:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}"
        params = {'localization': 'false', 'tickers': 'false', 'community_data': 'false', 'developer_data': 'false'}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            'name': data.get('name', symbol),
            'symbol': data.get('symbol', '').upper(),
            'description': data.get('description', {}).get('en', '')[:200] if data.get('description') else ''
        }
    except Exception as e:
        print(f"CoinGecko info error: {e}")
        return None
