# 📈 AntigravityStocks - AI-Powered Market Analytics Dashboard

A premium, real-time stock and cryptocurrency analysis dashboard with AI-powered predictions, built with React and Flask.

![Dashboard Preview](https://img.shields.io/badge/Status-Live-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![React](https://img.shields.io/badge/React-18+-61dafb)

## ✨ Features

### 🎯 Core Functionality
- **Real-Time Data**: Live stock and crypto prices with auto-refresh (30s intervals)
- **Dual Data Sources**: 
  - CoinGecko API for accurate crypto prices (20+ cryptocurrencies)
  - Yahoo Finance for stocks
- **Interactive Charts**: Beautiful Plotly visualizations with zoom, pan, and hover details
- **Technical Analysis**: RSI, SMA (20/50), volume analysis
- **AI Predictions**: Linear Regression and LSTM deep learning models

### 💎 Premium Features
- **Auto-Refresh**: Toggle on/off with visual indicators
- **Smart Fallback**: Automatic failover if API rate limits hit
- **Data Source Transparency**: Shows which API provided the data
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark Theme**: Eye-friendly glassmorphism UI with gradients

### 📊 Supported Assets

**Cryptocurrencies** (via CoinGecko):
- BTC, ETH, SOL, ADA, DOT, DOGE, MATIC, LINK, UNI, AVAX
- XRP, LTC, BCH, ATOM, XLM, ALGO, VET, FIL, TRX, ETC

**Stocks** (via yfinance):
- Any stock symbol: AAPL, TSLA, NVDA, GOOGL, MSFT, AMZN, etc.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
```bash
cd "c:/Users/BCC/Desktop/New folder (3)"
```

2. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Backend will run on `http://127.0.0.1:5000`

3. **Frontend Setup** (in a new terminal)
```bash
cd frontend
npm install
npm run dev
```
Frontend will run on `http://localhost:5173`

4. **Open Dashboard**
Navigate to `http://localhost:5173` in your browser

## 📖 Usage

### Search for Assets
- **Crypto**: Type `BTC`, `ETH`, `SOL` (auto-converts to proper format)
- **Stocks**: Type `AAPL`, `TSLA`, `NVDA`

### View Analytics
- **Current Price** with 24h change percentage
- **24h High/Low** prices
- **Volume** and **Open** price
- **RSI Indicator** (Overbought/Oversold signals)
- **Moving Averages** (SMA 20 & 50)

### AI Predictions
1. Search for any asset
2. Click **"Linear Regression"** for trend-based forecast
3. Click **"LSTM (Deep Learning)"** for neural network prediction
4. View 7-day price forecast on the chart

### Auto-Refresh
- Click **🔄 Auto-Refresh** to toggle automatic updates
- Updates every 30 seconds when enabled
- See **"Last updated"** timestamp in header

## 🏗️ Tech Stack

### Backend
- **Flask**: REST API framework
- **yfinance**: Stock market data
- **CoinGecko API**: Real-time crypto prices
- **Pandas & NumPy**: Data processing
- **Scikit-learn**: Machine learning (Linear Regression)
- **TensorFlow/Keras**: Deep learning (LSTM)

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool
- **Axios**: HTTP client
- **Plotly.js**: Interactive charts
- **Lucide React**: Icons
- **CSS Variables**: Custom theming

## 📁 Project Structure

```
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt       # Python dependencies
│   └── services/
│       ├── prediction.py      # ML/DL prediction models
│       └── coingecko.py       # CoinGecko API integration
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── index.css         # Global styles
│   │   └── components/
│   │       └── StockChart.jsx # Chart component
│   ├── index.html
│   └── package.json
└── README.md
```

## 🔧 API Endpoints

### Backend API

**Health Check**
```
GET /health
```

**Get Stock/Crypto Data**
```
GET /api/stock/<symbol>
Response: { symbol, company, data[], stats, data_source, warning }
```

**Get AI Prediction**
```
GET /api/predict/<symbol>?model=linear|lstm
Response: { symbol, model, predictions[] }
```

## 🎨 Features in Detail

### Data Sources
- **CoinGecko**: Real-time crypto prices, 24h stats, historical data
- **yfinance**: Stock prices, historical data, company info
- **Smart Routing**: Auto-detects crypto vs stock symbols

### Technical Indicators
- **RSI (14)**: Relative Strength Index for overbought/oversold signals
- **SMA 20**: 20-day Simple Moving Average
- **SMA 50**: 50-day Simple Moving Average

### AI Models
- **Linear Regression**: Fast, trend-following predictions
- **LSTM**: Deep learning for complex pattern recognition
- **7-Day Forecast**: Future price predictions with visual overlay

## 🔐 Rate Limits

### CoinGecko (Free Tier)
- ~10-50 calls/minute
- Auto-fallback to yfinance if exceeded
- Current price endpoint separate (faster)

### yfinance
- No strict rate limits
- May have delays for crypto data

## 🐛 Troubleshooting

**Backend won't start**
```bash
# Check if port 5000 is available
netstat -ano | findstr :5000

# Install dependencies
pip install -r requirements.txt
```

**Frontend won't start**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**No crypto data**
- Check if CoinGecko API is accessible
- System will auto-fallback to yfinance
- Check backend console for error messages

**LSTM predictions slow**
- TensorFlow installation required
- First prediction trains model (takes time)
- Subsequent predictions are faster

## 📝 Notes

- **TensorFlow**: Optional for LSTM predictions
- **Auto-Refresh**: Only works when dashboard is open
- **Data Accuracy**: CoinGecko provides most accurate crypto prices
- **Historical Data**: 1 year of data for analysis

## 🎯 Future Enhancements

- [ ] WebSocket for true real-time updates
- [ ] Multiple watchlists
- [ ] Price alerts and notifications
- [ ] More technical indicators (MACD, Bollinger Bands)
- [ ] Portfolio tracking
- [ ] News integration
- [ ] Social sentiment analysis

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **CoinGecko** for crypto market data
- **Yahoo Finance** for stock market data
- **Google Deepmind** for Antigravity AI assistant

---

**Built with ❤️ using React, Flask, and AI**

🚀 **Go to `http://localhost:5173` to start trading smarter!**
