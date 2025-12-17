import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    print("TensorFlow not installed. LSTM features will utilize a mock or be unavailable.")
    tf = None

def prepare_data(df, look_back=60):
    """
    Prepare data for LSTM.
    """
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, scaled_data

def train_linear_regression(df):
    """
    Train a simple Linear Regression model.
    Predicts next day based on numeric date.
    """
    df = df.copy()
    df['Date_Ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    
    X = df[['Date_Ordinal']]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def train_lstm_model(df):
    """
    Train a simple LSTM model.
    Falls back to MLPRegressor (Neural Net) if TensorFlow is not installed.
    """
    # Use MLPRegressor as fallback if TF is not available
    if tf is None:
        print("TensorFlow not found. Falling back to sklearn MLPRegressor (Lightweight Neural Net).")
        from sklearn.neural_network import MLPRegressor
        
        # Prepare data for MLP (Non-sequential input usually, but we can use window)
        # Re-using the same prepare_data format but flattening X
        x_train, y_train, scaler, scaled_data = prepare_data(df)
        
        # Flatten x_train for MLP: [samples, look_back, 1] -> [samples, look_back]
        nsamples, nx, ny = x_train.shape
        x_train_flat = x_train.reshape((nsamples, nx*ny))
        
        model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=200, random_state=42)
        model.fit(x_train_flat, y_train)
        
        return model, scaler

    # TensorFlow LSTM implementation
    x_train, y_train, scaler, scaled_data = prepare_data(df)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
    
    return model, scaler

def predict_future_linear(model, last_date, days=7):
    future_dates = []
    current_date = pd.to_datetime(last_date)
    
    for i in range(1, days + 1):
        future_dates.append(current_date + pd.Timedelta(days=i))
        
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predictions = model.predict(future_ordinals)
    
    return [{"date": d.strftime('%Y-%m-%d'), "price": p} for d, p in zip(future_dates, predictions)]

def predict_future_lstm(model, scaler, data, look_back=60, days=7):
    # Check if model is sklearn MLP or TF Sequential
    is_sklearn = False
    try:
        from sklearn.neural_network import MLPRegressor
        if isinstance(model, MLPRegressor):
            is_sklearn = True
    except:
        pass
        
    # data is the full scaled dataset
    curr_input = data[-look_back:].reshape(1, look_back, 1)
    predictions = []
    
    temp_input = list(curr_input[0])
    temp_input = [x[0] for x in temp_input] # flatten
    
    for i in range(days):
        if is_sklearn:
            # Flatten input for MLP
            x_input = np.array(temp_input[-look_back:]).reshape(1, look_back)
            pred = model.predict(x_input)
            # MLP returns [value], TF returns [[value]]
            pred_val = pred[0]
        else:
            # 3D input for LSTM
            x_input = np.array(temp_input[-look_back:]).reshape(1, look_back, 1)
            pred = model.predict(x_input, verbose=0)
            pred_val = pred[0][0]
            
        predictions.append(pred_val)
        temp_input.append(pred_val)
        
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten().tolist()

def calculate_trading_signals(df, strategy="day_trading"):
    """
    Generate Advanced Trading Signals with detailed technical analysis.
    Optimized for Day Trading (Faster indicators, tighter stops).
    """
    if df is None or df.empty or len(df) < 20:
        return None
        
    # Get latest data points
    latest = df.iloc[-1]
    
    close = float(latest['Close'])
    
    # --- 1. RSI Analysis (Standard 14) ---
    rsi = float(latest['RSI']) if 'RSI' in df.columns else 50.0
    rsi_signal = "NEUTRAL"
    # Day trading often uses slightly more extreme levels or rapid reversals, 
    # but 30/70 is still standard.
    if rsi < 30:
        rsi_signal = "BUY"
    elif rsi > 70:
        rsi_signal = "SELL"
        
    # --- 2. Moving Average Analysis (EMA 9 vs 21 for Speed) ---
    # We calculate EMAs on the fly as they are faster for day trading than SMA 20/50
    ema9 = df['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = df['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
    
    ma_signal = "NEUTRAL"
    if ema9 > ema21:
        ma_signal = "BUY" # Fast Uptrend
    elif ema9 < ema21:
        ma_signal = "SELL" # Fast Downtrend
        
    # --- 3. MACD Calculation ---
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    curr_macd = macd_line.iloc[-1]
    curr_sig = signal_line.iloc[-1]
    prev_macd = macd_line.iloc[-2]
    prev_sig = signal_line.iloc[-2]
    
    macd_signal = "NEUTRAL"
    if curr_macd > curr_sig and prev_macd <= prev_sig:
        macd_signal = "STRONG BUY"
    elif curr_macd < curr_sig and prev_macd >= prev_sig:
        macd_signal = "STRONG SELL"
    elif curr_macd > curr_sig:
        macd_signal = "BUY"
    elif curr_macd < curr_sig:
        macd_signal = "SELL"

    # --- 4. Volatility / Bollinger for Scalping ---
    sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
    std_dev = df['Close'].rolling(window=20).std().iloc[-1]
    upper_band = sma20 + (std_dev * 2)
    lower_band = sma20 - (std_dev * 2)
    
    bb_signal = "NEUTRAL"
    if close <= lower_band:
        bb_signal = "STRONG BUY" # Scalp Buy
    elif close >= upper_band:
        bb_signal = "STRONG SELL" # Scalp Sell

    # --- Overall Consensus ---
    score = 0
    signals_list = [rsi_signal, ma_signal, macd_signal, bb_signal]
    for s in signals_list:
        if "STRONG BUY" in s: score += 2
        elif "BUY" in s: score += 1
        elif "STRONG SELL" in s: score -= 2
        elif "SELL" in s: score -= 1
        
    overall_signal = "NEUTRAL"
    confidence = "Low"
    
    if score >= 3:
        overall_signal = "STRONG BUY"
        confidence = "High"
    elif score >= 1:
        overall_signal = "BUY"
        confidence = "Medium"
    elif score <= -3:
        overall_signal = "STRONG SELL"
        confidence = "High"
    elif score <= -1:
        overall_signal = "SELL"
        confidence = "Medium"
        
    # --- Day Trading Levels (Tighter) ---
    # Use recent swing highs/lows (last 5-10 candles) for tighter stop loss
    support = df['Low'].tail(10).min()
    resistance = df['High'].tail(10).max()
    
    # Calculate ATR (Average True Range) approx for dynamic SL/TP
    tr = df['High'] - df['Low']
    atr = tr.tail(14).mean()
    
    # --- Risk Management Strategy ---
    if strategy == "scalping_xau":
        # Specific Scalping Strategy for Gold (50-100 pips target)
        # Assuming 1 pip = 0.1 price change (standard), 50-100 pips = $5.0 - $10.0 move
        tp_target = 7.5 # Aim for ~75 pips ($7.50) avg
        sl_target = 4.0 # Risk ~40 pips ($4.00) avg
        
        if "BUY" in overall_signal:
            sl = close - sl_target
            tp = close + tp_target
        elif "SELL" in overall_signal:
            sl = close + sl_target
            tp = close - tp_target
        else:
            sl = close - sl_target
            tp = close + sl_target
            
    else:
        # Standard Day Trading Risk Management: 1:1.5 or 1:2 Risk Reward, based on volatility (ATR)
        if "BUY" in overall_signal:
            # Stop Loss below recent support or 1.5 ATR
            sl = support if (close - support) < (2 * atr) else close - (1.5 * atr)
            risk = close - sl
            if risk <= 0: risk = atr * 0.5 # fallback
            tp = close + (risk * 2) # Target 1:2
            
        elif "SELL" in overall_signal:
            sl = resistance if (resistance - close) < (2 * atr) else close + (1.5 * atr)
            risk = sl - close
            if risk <= 0: risk = atr * 0.5
            tp = close - (risk * 2)
            
        else:
            # Neutral choppy market
            sl = close - atr
            tp = close + atr

    return {
        "signal": overall_signal,
        "confidence": confidence,
        "score": score,
        "entry_price": close,
        "stop_loss": sl,
        "take_profit": tp,
        "strategy": "Day Trading (Intraday)",
        "analysis": [
            { "name": "RSI (14)", "value": f"{rsi:.2f}", "signal": rsi_signal, "condition": "Momentum (<30 Buy, >70 Sell)" },
            { "name": "MACD", "value": f"{curr_macd:.2f}", "signal": macd_signal, "condition": "Trend Crossover" },
            { "name": "EMA Trend (9 vs 21)", "value": "Bullish" if ma_signal == "BUY" else "Bearish", "signal": ma_signal, "condition": "Fast Moving Averages" },
            { "name": "Bollinger Bands", "value": "Volatility", "signal": bb_signal, "condition": "Reversion (Outer Bands)" }
        ]
    }
