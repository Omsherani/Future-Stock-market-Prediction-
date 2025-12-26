import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# --------------------
# Health Check
# --------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Stock Prediction API is running"
    })

# --------------------
# Internal Imports
# --------------------
from services.prediction import (
    train_linear_regression,
    predict_future_linear,
    train_lstm_model,
    predict_future_lstm,
    calculate_trading_signals
)

from services.coingecko import (
    is_crypto_symbol,
    fetch_crypto_historical_data,
    fetch_crypto_current_price,
    get_crypto_info
)

from services.binance_api import (
    get_binance_klines,
    get_binance_price,
    get_binance_24hr_stats
)

# Optional TradingView import (prevents Render crashes)
try:
    from tradingview_ta import TA_Handler, Interval
    TRADINGVIEW_AVAILABLE = True
except Exception:
    TRADINGVIEW_AVAILABLE = False
    TA_Handler = None
    Interval = None


@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        symbol = symbol.upper()
        original_symbol = symbol

        is_crypto = is_crypto_symbol(symbol)

        # =========================
        # CRYPTO / GOLD
        # =========================
        if is_crypto or symbol in ('XAUUSD', 'GOLD'):
            current_data = None
            crypto_info = None
            data_source = None

            # -------- GOLD (XAUUSD) --------
            if symbol in ('XAUUSD', 'GOLD'):
                crypto_info = {'name': 'Gold Spot (XAU/USD)'}

                # Binance PAXG (Primary)
                try:
                    paxg_price = get_binance_price("PAXGUSDT")
                    if paxg_price:
                        current_data = {
                            "price": paxg_price,
                            "change_24h": 0,
                            "volume_24h": 0
                        }
                        data_source = "Binance PAXG"
                except Exception:
                    pass

                # TradingView (Optional)
                if not current_data and TRADINGVIEW_AVAILABLE:
                    tv_sources = [
                        {"exchange": "OANDA", "screener": "forex"},
                        {"exchange": "FX_IDC", "screener": "forex"},
                        {"exchange": "FXCM", "screener": "forex"},
                    ]
                    for cfg in tv_sources:
                        try:
                            handler = TA_Handler(
                                symbol="XAUUSD",
                                screener=cfg["screener"],
                                exchange=cfg["exchange"],
                                interval=Interval.INTERVAL_1_MINUTE
                            )
                            price = handler.get_analysis().indicators.get("close")
                            if price:
                                current_data = {
                                    "price": price,
                                    "change_24h": 0,
                                    "volume_24h": 0
                                }
                                data_source = f"TradingView {cfg['exchange']}"
                                break
                        except Exception:
                            continue

                # yfinance fallback (IAU proxy)
                stock = yf.Ticker("IAU")
                hist = stock.history(period="1y")
                if hist.empty:
                    return jsonify({"error": "No gold data available"}), 404

                hist.reset_index(inplace=True)
                scale_factor = 53.4
                for col in ['Open', 'High', 'Low', 'Close']:
                    hist[col] *= scale_factor

                if not current_data:
                    current_data = {
                        "price": float(hist['Close'].iloc[-1]),
                        "change_24h": 0,
                        "volume_24h": 0
                    }
                    data_source = "yfinance IAU (Scaled)"

            # -------- CRYPTO --------
            else:
                stats_24h = get_binance_24hr_stats(symbol)
                hist = get_binance_klines(symbol)

                if hist is not None and not hist.empty:
                    data_source = "Binance API"
                    crypto_info = {"name": f"{symbol}/USDT"}

                    if stats_24h:
                        current_data = {
                            "price": stats_24h["price"],
                            "change_24h": stats_24h["change_24h"],
                            "volume_24h": stats_24h["volume_24h"]
                        }
                else:
                    current_data = fetch_crypto_current_price(symbol)
                    crypto_info = get_crypto_info(symbol)
                    hist = fetch_crypto_historical_data(symbol, days=365)
                    data_source = "CoinGecko"

            if hist is None or hist.empty:
                return jsonify({"error": "No data found"}), 404

            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')

        # =========================
        # STOCKS
        # =========================
        else:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            if hist.empty:
                return jsonify({"error": "No stock data found"}), 404

            hist.reset_index(inplace=True)
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            data_source = "yfinance"
            crypto_info = {"name": symbol}

        # =========================
        # INDICATORS
        # =========================
        hist['SMA_20'] = hist['Close'].rolling(20).mean()
        hist['SMA_50'] = hist['Close'].rolling(50).mean()

        delta = hist['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        df = hist.dropna()
        data = [{
            "Date": r['Date'],
            "Open": float(r['Open']),
            "High": float(r['High']),
            "Low": float(r['Low']),
            "Close": float(r['Close']),
            "Volume": int(r['Volume']),
            "SMA_20": float(r['SMA_20']),
            "SMA_50": float(r['SMA_50']),
            "RSI": float(r['RSI'])
        } for _, r in df.iterrows()]

        latest = df.iloc[-1]
        stats = {
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "close": float(latest['Close']),
            "volume": int(latest['Volume'])
        }

        signals = calculate_trading_signals(hist)

        return jsonify({
            "symbol": symbol,
            "company": crypto_info["name"],
            "data": data,
            "stats": stats,
            "signals": signals,
            "data_source": data_source,
            "warning": None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# ENTRY POINT (Render)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
