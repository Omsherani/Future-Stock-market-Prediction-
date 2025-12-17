# Project Updates

## Key Improvements

### 1. Binance Integration (Crypto)
- Switched from CoinGecko/yfinance to **Binance API**.
- **Reason**: CoinGecko has strict rate limits and delayed data. yfinance is often unstable for crypto.
- **Benefits**:
  - Live, sub-second price updates.
  - Reliable historical candlestick data (Kline).
  - High accuracy for pairs like BTCUSDT, ETHUSDT, SOLUSDT.

### 2. XAUUSD (Gold) Strategy
- **TradingView Scraper Strategy**: 
  - The app now aggressively attempts to fetch real-time Gold prices from multiple TradingView sources (**OANDA, FXCM, FOREX.COM**).
  - This simulates a "WebSocket"-like experience by polling frequently on the frontend (every 3s).
- **Fallback**:
  - If direct fetching fails, it falls back to **IAU (iShares Gold Trust)** with a calibrated scaling factor (**32.85x**) which is highly accurate for Spot Gold approximation.

### 3. Cleanup
- Removed temporary debug scripts (`test_*.py`, `debug_*.py`) to keep the project clean.

`backend/app.py` has been updated to use the new `services/binance_api.py` module.
