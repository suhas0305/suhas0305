import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Step 1: Data Collection
# Fetch historical stock price data for Adani Group (ADANIGREEN.BO) from Yahoo Finance
ticker = "ADANIGREEN.BO"
start_date = "2010-01-01"
end_date = "2022-01-01"
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Data Preprocessing
# Extract 'Close' prices
close_prices = stock_data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 60  # Length of sequences
X, y = create_sequences(scaled_data, sequence_length)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 3: Model Building
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 4: Model Evaluation
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Step 5: Prediction
# Predict stock prices for the next 'n' days
future_days = 30
future_dates = pd.date_range(start=end_date, periods=future_days)
future_prices = []

last_sequence = scaled_data[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, 1)

for _ in range(future_days):
    prediction = model.predict(last_sequence)
    future_prices.append(prediction[0])
    last_sequence = np.append(last_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# Inverse transform the predicted prices
predicted_prices = scaler.inverse_transform(future_prices)

# Display the predicted prices
future_prices_df = pd.DataFrame(predicted_prices, index=future_dates, columns=[f'Predicted {ticker} Price'])
print(future_prices_df)

# Plot the predicted prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[-100:], stock_data['Close'].values[-100:], label='Actual Stock Prices')
plt.plot(future_dates, predicted_prices, label='Predicted Stock Prices', linestyle='--')
plt.title('Adani Group Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.legend()
plt.show()