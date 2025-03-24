import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime

# Configuration
TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'TSLA', 'INFY', 'TM']  # List of stock tickers  # List of stock tickers
LOOKBACK_DAYS = 60  # Number of days to look back for training
START_DATE = '2015-01-01'  # Historical data start date

# Create necessary directories
os.makedirs('stock_models', exist_ok=True)

def download_stock_data(ticker):
    """Downloads historical stock data using Yahoo Finance API."""
    try:
        df = yf.download(ticker, start=START_DATE)
        if df.empty:
            raise ValueError("No data downloaded")
        return df
    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return None

def preprocess_data(df):
    """Scales the 'Close' prices using MinMaxScaler and prepares sequences."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    
    x_train, y_train = [], []
    for i in range(LOOKBACK_DAYS, len(scaled_data)):
        x_train.append(scaled_data[i-LOOKBACK_DAYS:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

def build_lstm_model():
    """Creates and compiles an LSTM model for stock price prediction."""
    model = Sequential([
        Input(shape=(LOOKBACK_DAYS, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        
        # Step 1: Download stock data
        stock_data = download_stock_data(ticker)
        if stock_data is None:
            continue
        
        # Step 2: Preprocess data
        x_train, y_train, _ = preprocess_data(stock_data)
        
        # Step 3: Build and train the model
        model = build_lstm_model()
        model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1)
        
        # Step 4: Save the trained model
        model_path = f'stock_models/{ticker}_model.keras'
        save_model(model, model_path)
        print(f"Model saved: {model_path}")
        print("=" * 60)
