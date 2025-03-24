import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from datetime import datetime

# Configuration
TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'TSLA', 'INFY', 'TM']  # List of stock tickers
LOOKBACK_DAYS = 60  # Must match training lookback period
TRAIN_END_DATE = '2023-06-01'  # Training data cutoff
TEST_START_DATE = '2023-07-01'  # Validation period start

def load_stock_data(ticker):
    """Downloads historical stock data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start='2015-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        if df.empty:
            raise ValueError("No data available")
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def validate_model(ticker):
    """Loads a trained model, makes predictions, and evaluates performance."""
    try:
        # Load the trained LSTM model
        model_path = f'stock_models/{ticker}_model.keras'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path)
        
        # Load stock data
        df = load_stock_data(ticker)
        if df is None:
            return
        
        # Split into training and testing sets
        train_data = df[:TRAIN_END_DATE]
        test_data = df[TEST_START_DATE:]
        
        # Scale the data using training data only
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data[['Close']])
        
        # Make rolling predictions
        predictions, actuals, dates = [], [], []
        for i in range(len(test_data)):
            combined_data = pd.concat([train_data, test_data.iloc[:i]])
            recent_data = combined_data[-LOOKBACK_DAYS:]
            
            if len(recent_data) < LOOKBACK_DAYS:
                continue  # Skip incomplete windows
            
            scaled_window = scaler.transform(recent_data[['Close']])
            model_input = scaled_window.reshape(1, LOOKBACK_DAYS, 1)
            prediction = scaler.inverse_transform(model.predict(model_input))[0][0]
            
            predictions.append(prediction)
            actuals.append(test_data.iloc[i]['Close'])
            dates.append(test_data.index[i])
        
        # Compute error metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        # Plot results
        plt.figure(figsize=(16, 8))
        plt.plot(df.index, df['Close'], label='Historical Prices', alpha=0.5)
        plt.plot(dates, predictions, label='Predictions', linestyle='--')
        plt.scatter(dates, actuals, color='green', s=20, label='Actual Prices')

        plt.title(f"{ticker} Model Performance: {TEST_START_DATE} to {datetime.today().strftime('%Y-%m-%d')}")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.axvline(pd.to_datetime(TRAIN_END_DATE), color='r', linestyle='--', label='Train/Test Split')
        plt.legend()
        plt.grid(True)

        # Add metrics annotation
        plt.annotate(f"MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}",
                    xy=(0.05, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round", alpha=0.1))

        plt.savefig(f'prediction_graphs/{ticker}_validation.png', bbox_inches='tight')
        plt.close()
        
        # Print validation results
        print(f"\n{ticker} Model Validation:")
        print(f"Date Range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print("=" * 60)
    
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        print("=" * 60)

if __name__ == "__main__":
    os.makedirs('prediction_graphs', exist_ok=True)
    for ticker in TICKERS:
        validate_model(ticker)

