import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap

# Streamlit configuration
st.set_page_config(layout="wide")

# Function to download stock data
@st.cache_data
def download_data(tickers, period="3mo", interval="1d"):
    try:
        # Ensure tickers is a list
        if isinstance(tickers, str):
            tickers = [tickers]

        # Download data with group_by to handle multiple tickers
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker")
        
        if len(tickers) == 1:  # Flatten if single ticker
            ticker = tickers[0]
            data.columns = pd.MultiIndex.from_product([[ticker], data.columns])

        data.ffill(inplace=True)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Calculate SMA
def calculate_sma(data, tickers):
    for ticker in tickers:
        if ticker in data.columns.get_level_values(0):
            data[(ticker, 'SMA_30')] = data[(ticker, 'Adj Close')].rolling(window=30, min_periods=1).mean()
            data[(ticker, 'SMA_60')] = data[(ticker, 'Adj Close')].rolling(window=60, min_periods=1).mean()
    return data

# SMA strategy
def sma_strategy(data, ticker):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals.loc[data[(ticker, 'SMA_30')] > data[(ticker, 'SMA_60')], 'signal'] = 1.0
    signals['positions'] = signals['signal'].diff()
    return signals

# Prepare dataset for LSTM
def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Train LSTM model
def train_lstm(X_train, y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    return model

# Calculate SHAP values
def calculate_shap(model, X_lstm):
    sample_size = min(100, len(X_lstm))
    background = X_lstm[:sample_size].reshape(sample_size, -1)
    explainer = shap.KernelExplainer(lambda x: model.predict(x.reshape(-1, X_lstm.shape[1], 1)), background)

    num_samples = min(10, len(X_lstm))
    sample_data = X_lstm[:num_samples].reshape(num_samples, -1)
    shap_values = explainer.shap_values(sample_data)
    return shap_values, sample_data

# Visualization functions
def plot_shap(shap_values, X_sample, feature_names):
    fig, ax = plt.subplots()
    shap.summary_plot(np.array(shap_values).squeeze(), features=X_sample, feature_names=feature_names, plot_type="dot", show=False)
    st.pyplot(fig)
    plt.close(fig)

def plot_stock_signals(data, ticker):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(data.index, data[(ticker, 'Adj Close')], label=f"{ticker} Close", color='blue', linewidth=1.5)
    ax.plot(data.index, data[(ticker, 'SMA_30')], label="SMA Short (30)", color='red', linestyle='--', linewidth=2)
    ax.plot(data.index, data[(ticker, 'SMA_60')], label="SMA Long (60)", color='green', linestyle='--', linewidth=2)

    signals = sma_strategy(data, ticker)
    buy_signals = signals.positions == 1.0
    sell_signals = signals.positions == -1.0
    ax.plot(data.index[buy_signals], data[(ticker, 'Adj Close')][buy_signals], '^', color='green', label='Buy Signal', markersize=10)
    ax.plot(data.index[sell_signals], data[(ticker, 'Adj Close')][sell_signals], 'v', color='red', label='Sell Signal', markersize=10)

    ax.set_title(f"{ticker} Stock Price and SMA Strategy", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    plt.close(fig)

# Streamlit app layout
st.title("Comprehensive Stock Analysis Dashboard")

# Dynamic input for stock tickers
tickers_input = st.text_input("Enter Stock Tickers (comma-separated)", value="AAPL")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Ensure at least one valid ticker is provided
if not tickers:
    st.warning("Please enter at least one valid stock ticker.")
else:
    period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Select Data Interval", ["1d", "1wk"], index=0)

    data = download_data(tickers, period, interval)

    if data is not None:
        data = calculate_sma(data, tickers)

        selected_ticker = st.selectbox("Select a Stock for Detailed Analysis", tickers)

        if selected_ticker:
            plot_stock_signals(data, selected_ticker)

            df = data[(selected_ticker, 'Adj Close')].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df)

            if len(df_scaled) > 50:
                X, Y = create_dataset(df_scaled, 10)
                X_lstm = X.reshape((X.shape[0], X.shape[1], 1))

                model = train_lstm(X_lstm, Y)

                # Evaluate model
                predictions = model.predict(X_lstm)
                rmse = np.sqrt(mean_squared_error(Y, predictions))
                mae = mean_absolute_error(Y, predictions)
                st.write(f"Model Evaluation: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

                shap_values, X_sample = calculate_shap(model, X_lstm)

                feature_names = [f"Day {i+1}" for i in range(10)]
                st.subheader("SHAP Value Analysis (Explainability)")
                plot_shap(shap_values, X_sample, feature_names)
            else:
                st.warning("Not enough data to train the LSTM model.")
    else:
        st.warning("Unable to fetch data. Please check your tickers or settings.")
