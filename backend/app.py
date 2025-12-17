from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Stock Prediction API is running"})

from services.prediction import train_linear_regression, predict_future_linear, train_lstm_model, predict_future_lstm, calculate_trading_signals
from services.coingecko import is_crypto_symbol, fetch_crypto_historical_data, fetch_crypto_current_price, get_crypto_info
from services.binance_api import get_binance_klines, get_binance_price, get_binance_24hr_stats

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        symbol = symbol.upper()
        original_symbol = symbol
        
        # Check if XAUUSD or Crypto
        is_crypto = is_crypto_symbol(symbol)
        
        if is_crypto or symbol == 'XAUUSD' or symbol == 'GOLD':
            print(f"Fetching data for {symbol}...")
            
            # For XAUUSD/GOLD, use PAXGUSDT (Pax Gold) from Binance as primary 24/7 source
            if symbol == 'XAUUSD' or symbol == 'GOLD':
                print("Fetching XAUUSD (using PAXGUSDT as proxy)...")
                current_data = None
                crypto_info = {'name': 'Gold Spot (XAU/USD)'}
                
                # 1. Try Binance PAXGUSDT (Most reliable 24/7 source)
                # PAXG tracks 1 oz fine gold
                try:
                    paxg_price = get_binance_price("PAXGUSDT")
                    if paxg_price:
                        print(f"✓ Binance PAXG Price: ${paxg_price:.2f}")
                        current_data = {
                            'price': paxg_price,
                            'change_24h': 0,
                            'volume_24h': 0
                        }
                        data_source = "Binance PAXG (Gold Proxy)"
                except Exception as e:
                    print(f"Binance PAXG fetch failed: {e}")

                # 2. If PAXG failed, try TradingView
                if not current_data:
                    try:
                        from tradingview_ta import TA_Handler, Interval
                        # ... (existing TV code) ...
                        tv_configs = [
                            {"exchange": "OANDA", "screener": "forex", "symbol": "XAUUSD"},
                            {"exchange": "FX_IDC", "screener": "forex", "symbol": "XAUUSD"},
                            {"exchange": "FXCM", "screener": "forex", "symbol": "XAUUSD"}
                        ]
                        for cfg in tv_configs:
                             # ... simplified loop ...
                             try:
                                handler = TA_Handler(symbol=cfg['symbol'], screener=cfg['screener'], exchange=cfg['exchange'], interval=Interval.INTERVAL_1_MINUTE)
                                price = handler.get_analysis().indicators.get('close')
                                if price:
                                     current_data = {'price': price, 'change_24h': 0, 'volume_24h': 0}
                                     print(f"✓ TV Price: {price}")
                                     data_source = f"TradingView {cfg['exchange']}"
                                     break
                             except:
                                 continue
                    except Exception as e:
                        print(f"TV fetch failed: {e}")

                # 3. Fallback to IAU (re-calibrated for Dec 2025)
                # IAU ~$81, Gold ~$4330 -> Factor ~53.4
                scale_factor = 53.4
                stock = yf.Ticker('IAU')
                try:
                    hist = stock.history(period="1y")
                    if not hist.empty:
                        hist.reset_index(inplace=True)
                        hist['Open'] *= scale_factor
                        hist['High'] *= scale_factor
                        hist['Low'] *= scale_factor
                        hist['Close'] *= scale_factor
                        
                        if current_data is None:
                            latest_close = hist['Close'].iloc[-1]
                            print(f"Using IAU scaled price: ${latest_close:.2f}")
                            current_data = {
                                'price': float(latest_close),
                                'change_24h': 0,
                                'volume_24h': 0
                            }
                            data_source = "yfinance IAU (Scaled)"
                    else:
                        return jsonify({"error": "No gold data available"}), 404
                except Exception as e:
                     return jsonify({"error": str(e)}), 500
                

            
            else:
                # For other crypto, use Binance API (Much faster and cleaner)
                print(f"Fetching crypto data for {symbol} from Binance...")
                
                # Fetch 24hr stats for current price/info
                stats_24h = get_binance_24hr_stats(symbol)
                
                # Fetch Hist Data
                hist = get_binance_klines(symbol)
                
                if hist is not None and not hist.empty:
                    data_source = "Binance API (Live)"
                    
                    if stats_24h:
                        current_data = {
                            'price': stats_24h['price'],
                            'change_24h': stats_24h['change_24h'],
                            'volume_24h': stats_24h['volume_24h'] # volume in base asset
                        }
                        crypto_info = {'name': f"{symbol}/USDT"}
                    else:
                         current_data = None
                         crypto_info = {'name': symbol}
                         
                else:
                    # Fallback to CoinGecko if Binance fails (e.g. unknown symbol on Binance)
                    print("Binance failed, falling back to CoinGecko...")
                    current_data = fetch_crypto_current_price(symbol)
                    crypto_info = get_crypto_info(symbol)
                    hist = fetch_crypto_historical_data(symbol, days=365)
                    data_source = "CoinGecko (Fallback)"

            if hist is None or hist.empty:
                 # Last resort YFinance
                 return jsonify({"error": f"No crypto data found for {symbol}"}), 404
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            
            # Calculate technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            
            # RSI Calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Format for frontend
            df_clean = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']].dropna()
            
            data = []
            for _, row in df_clean.iterrows():
                data.append({
                    'Date': row['Date'],
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                    'Volume': int(row['Volume']) if row['Volume'] > 0 else 0,
                    'SMA_20': float(row['SMA_20']),
                    'SMA_50': float(row['SMA_50']),
                    'RSI': float(row['RSI'])
                })
            
            # Get latest stats - use real-time data if available
            if current_data:
                stats = {
                    "open": float(current_data['price']),  # CoinGecko free tier doesn't have open
                    "high": float(current_data['price'] * 1.01),  # Approximate
                    "low": float(current_data['price'] * 0.99),   # Approximate
                    "close": float(current_data['price']),
                    "volume": int(current_data.get('volume_24h', 0)),
                    "change_24h": float(current_data.get('change_24h', 0))
                }
            else:
                latest = hist.iloc[-1]
                stats = {
                    "open": float(latest['Open']),
                    "high": float(latest['High']),
                    "low": float(latest['Low']),
                    "close": float(latest['Close']),
                    "volume": int(latest['Volume'])
                }
            
            company_name = crypto_info['name'] if crypto_info else symbol
            
            # Fetch Intraday Data (1 day) specifically for Day Trading Signals
            print(f"Fetching intraday data for {symbol} signals...")
            hist_intraday = fetch_crypto_historical_data(symbol, days=1)
            
            # Sync Analysis Data with Real-Time TV Price
            active_hist = hist_intraday if (hist_intraday is not None and not hist_intraday.empty and len(hist_intraday) > 10) else hist
            
            # If we have a real-time price from TradingView/CoinGecko, ensure the latest candle reflects it
            if active_hist is not None and not active_hist.empty and current_data and 'price' in current_data:
                # Update last close to match the real-time ticker
                active_hist.loc[active_hist.index[-1], 'Close'] = float(current_data['price'])
                # Adjust high/low if current price is outside range
                if float(current_data['price']) > active_hist.loc[active_hist.index[-1], 'High']:
                     active_hist.loc[active_hist.index[-1], 'High'] = float(current_data['price'])
                if float(current_data['price']) < active_hist.loc[active_hist.index[-1], 'Low']:
                     active_hist.loc[active_hist.index[-1], 'Low'] = float(current_data['price'])

            # Determine Strategy
            strat = "day_trading"
            if symbol == 'XAUUSD' or symbol == 'GOLD':
                strat = "scalping_xau"

            signals = calculate_trading_signals(active_hist, strategy=strat)
            
            return jsonify({
                "symbol": symbol.upper(),
                "company": company_name,
                "data": data,
                "stats": stats,
                "signals": signals,
                "data_source": data_source,
                "warning": None
            })
        
        # For stocks, use yfinance
        else:
            print(f"Fetching stock data for {symbol} from yfinance...")
            stock = yf.Ticker(symbol)
            
            # Try to fetch data with timeout handling
            hist = None
            try:
                hist = stock.history(period="1y")
            except Exception as e:
                print(f"Error fetching 1y data: {e}")
            
            if hist is None or hist.empty:
                return jsonify({"error": f"No data found for symbol {symbol}"}), 404
            

            
        # Basic processing
        hist.reset_index(inplace=True)
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        
        # Calculate simplae SMA (Technical Indicator)
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        
        # RSI Calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        # Format for frontend - convert to native Python types to avoid JSON serialization issues
        df_clean = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']].dropna()
        
        # Convert all numeric columns to native Python types
        data = []
        for _, row in df_clean.iterrows():
            data.append({
                'Date': row['Date'],
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': float(row['Close']),
                'Volume': int(row['Volume']),
                'SMA_20': float(row['SMA_20']),
                'SMA_50': float(row['SMA_50']),
                'RSI': float(row['RSI'])
            })
        
        # Get latest stats for the dashboard
        latest = hist.iloc[-1]
        stats = {
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "close": float(latest['Close']),
            "volume": int(latest['Volume'])
        }

        company_name = symbol.upper()
        if company_name == "GLD":
             company_name = original_symbol
        # Optimize Name Fetching: stock.info is VERY slow.
        data_source = "yfinance"
        warning = None
        
        # Check if this might be crypto and add warning about accuracy
        if "-USD" in symbol:
            warning = "Crypto prices from yfinance may have delays. For real-time crypto prices, consider using dedicated crypto APIs like CoinGecko or Binance."
        
        try:
            # Try fast_info first (newer yfinance versions)
            if hasattr(stock, 'fast_info'):
                # fast_info is a dictionary-like object but keys might vary
                # usually just relying on symbol is safer for speed
                pass
        except:
            pass
        
        # Custom Intraday Fetch for Stocks 
        try:
             # Fetch 5 days to ensure enough data for indicators, 15m intervals
             stock_intraday = stock.history(period="5d", interval="15m")
             if not stock_intraday.empty:

                     
                 signals = calculate_trading_signals(stock_intraday)
             else:
                 signals = calculate_trading_signals(hist)
        except Exception as e:
            print(f"Intraday stock fetch failed: {e}")
            signals = calculate_trading_signals(hist)
        
        return jsonify({
            "symbol": symbol.upper(),
            "company": company_name,
            "data": data,
            "stats": stats,
            "signals": signals,
            "data_source": data_source,
            "warning": warning
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/price/<symbol>', methods=['GET'])
def get_current_price(symbol):
    """
    Lightweight endpoint to fetch ONLY the current price.
    Much faster for polling specific updates.
    """
    try:
        symbol = symbol.upper()
        
        # 1. XAUUSD / GOLD
        if symbol == 'XAUUSD' or symbol == 'GOLD':
            # 1. Try Binance PAXG
            try:
                price = get_binance_price("PAXGUSDT")
                if price:
                    return jsonify({"symbol": symbol, "price": price, "source": "Binance PAXG"})
            except:
                pass
            
            # 2. Try TradingView
            from tradingview_ta import TA_Handler, Interval
            tv_configs = [
                 {"exchange": "OANDA", "screener": "forex", "symbol": "XAUUSD"},
                 {"exchange": "FX_IDC", "screener": "forex", "symbol": "XAUUSD"},
                 {"exchange": "FXCM", "screener": "forex", "symbol": "XAUUSD"}
            ]
            for cfg in tv_configs:
                try:
                    handler = TA_Handler(symbol=cfg['symbol'], screener=cfg['screener'], exchange=cfg['exchange'], interval=Interval.INTERVAL_1_MINUTE) # type: ignore
                    price = handler.get_analysis().indicators.get('close')
                    if price:
                         return jsonify({"symbol": symbol, "price": price, "source": f"TradingView ({cfg['exchange']})"})
                except:
                    continue
            
            # 3. Fallback to IAU (Dec 2025 Factor ~53.4)
            try:
                stock = yf.Ticker('IAU')
                price = stock.history(period="1d")['Close'].iloc[-1] * 53.4
                return jsonify({"symbol": symbol, "price": price, "source": "IAU (Fallback)"})
            except:
                return jsonify({"error": "Could not fetch gold price"}), 500

        # 2. Crypto (Binance)
        elif is_crypto_symbol(symbol):
            price = get_binance_price(symbol)
            if price:
                return jsonify({"symbol": symbol, "price": price, "source": "Binance"})
            else:
                # Fallback
                data = fetch_crypto_current_price(symbol)
                if data:
                    return jsonify({"symbol": symbol, "price": data['price'], "source": "CoinGecko"})
        
        # 3. Stocks (yfinance)
        else:
            stock = yf.Ticker(symbol)
            # Try fast_info for speed
            if hasattr(stock, 'fast_info'):
                try:
                    price = stock.fast_info['last_price']
                    if price:
                        return jsonify({"symbol": symbol, "price": price, "source": "yfinance fast_info"})
                except:
                    pass
            
            # Fallback to regular history
            hist = stock.history(period="1d")
            if not hist.empty:
                return jsonify({"symbol": symbol, "price": hist['Close'].iloc[-1], "source": "yfinance history"})

        return jsonify({"error": "Price not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    try:
        model_type = request.args.get('model', 'linear')
        symbol = symbol.upper()
        

        
        hist = None
        
        # Handle XAUUSD/GOLD specially
        if symbol == 'XAUUSD' or symbol == 'GOLD':
            print(f"Fetching gold data for prediction: {symbol}")
            stock = yf.Ticker('IAU')
            try:
                hist = stock.history(period="1y")
                if not hist.empty:
                    hist.reset_index(inplace=True)
                    # Scale IAU by 32.85x to get gold spot price
                    hist['Open'] = hist['Open'] * 32.85
                    hist['High'] = hist['High'] * 32.85
                    hist['Low'] = hist['Low'] * 32.85
                    hist['Close'] = hist['Close'] * 32.85
                else:
                    print("IAU returned empty data for prediction")
            except Exception as e:
                print(f"Error fetching IAU for prediction: {e}")
        
        # Use Binance/CoinGecko for crypto
        elif is_crypto_symbol(symbol):
            print(f"Fetching crypto data for prediction: {symbol}")
            # Try Binance first
            hist = get_binance_klines(symbol, limit=365)
            
            if hist is None or hist.empty:
                print("Binance prediction fetch failed, trying CoinGecko...")
                hist = fetch_crypto_historical_data(symbol, days=365)
                if hist is not None and not hist.empty:
                    hist.reset_index(drop=True, inplace=True)
                else:
                    # Fallback to yfinance if CoinGecko fails
                    print(f"CoinGecko prediction fetch failed for {symbol}, falling back to yfinance")
                    stock = yf.Ticker(f"{symbol}-USD")
                    try:
                        hist = stock.history(period="1y")
                        if not hist.empty:
                            hist.reset_index(inplace=True)
                    except Exception as e:
                        print(f"yfinance fallback for prediction failed: {e}")

        else:
            # Use yfinance for stocks
            stock = yf.Ticker(symbol)
            try:
                hist = stock.history(period="1y")
                if not hist.empty:
                    hist.reset_index(inplace=True)
            except Exception as e:
                print(f"Error fetching data for prediction: {e}")
        
        if hist is None or hist.empty:
            return jsonify({"error": "No data found"}), 404
            

        
        predictions = []
        
        if model_type == 'linear':
            model = train_linear_regression(hist)
            predictions = predict_future_linear(model, hist['Date'].iloc[-1], days=7)
        elif model_type == 'lstm':
            # LSTM takes longer, just doing a quick demo version
            model, scaler = train_lstm_model(hist)
            # Need to re-prepare full data for prediction context
            # This is simplified for the demo
            from services.prediction import prepare_data
            _, _, _, scaled_data = prepare_data(hist)
            preds = predict_future_lstm(model, scaler, scaled_data, days=7)
            
            # generating dates
            last_date = hist['Date'].iloc[-1]
            future_dates = []
            for i in range(1, 8):
                future_dates.append(last_date + pd.Timedelta(days=i))
            
            predictions = [{"date": d.strftime('%Y-%m-%d'), "price": float(p)} for d, p in zip(future_dates, preds)]
            
        return jsonify({
            "symbol": symbol,
            "model": model_type,
            "predictions": predictions
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
