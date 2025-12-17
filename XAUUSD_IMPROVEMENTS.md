# XAUUSD Real-Time & Accuracy Improvements

## Changes Implemented

### 1. Frontend (App.jsx)
- **Removed Manual Refresh Button**: The dashboard now updates automatically.
- **Enabled Auto-Polling**: The app now checks for price updates every **3 seconds** to simulate a live feed.
- **Visual Feedback**: The price card flashes green/red on update to indicate activity.

### 2. Backend (app.py)
- **Enhanced Data Source Selection**:
  - Now attempts to fetch real-time XAUUSD data from multiple TradingView exchanges: **FX_IDC, OANDA, FXCM, FOREX.COM**.
  - This increases the chance of finding a working live feed.
- **Calibrated Fallback Price**:
  - If real-time fetch fails, the system falls back to `IAU` (iShares Gold Trust).
  - **Updated Scaling Factor**: Changed from `32.0` to **`32.85`** based on current market rates (Spot ~$2660 vs IAU ~$81). This reduces the price discrepancy significantly.

## How to Test
1. Observe the "Live Price" on the dashboard.
2. It should update every ~3 seconds.
3. Compare the price with a live chart (e.g. TradingView). The deviation should be minimal.
