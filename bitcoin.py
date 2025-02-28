import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

# Set the title for the Streamlit app
st.title("Bitcoin Price Prediction using LSTM")
st.write("This application predicts future Bitcoin prices for up to 30 days using an LSTM model.")

# Load Bitcoin data
# @st.cache
def load_data():
    data = yf.download('BTC-USD', start='2023-01-01', end='2024-11-05')
    data.reset_index(inplace=True)
    return data

data = load_data()

# Display last few rows of data
st.subheader("Bitcoin Prices from Jan 2023 to Sep 2024")
st.write(data)

# Plot historical prices
st.subheader("Historical Bitcoin Price")
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Close'], label='Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price in USD')
ax.legend()
st.pyplot(fig)

# Preprocess Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split data into training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Function to create dataset with look_back
def create_dataset(data, look_back=60):
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

look_back = 60
x_train, y_train = create_dataset(train_data, look_back)
x_test, y_test = create_dataset(test_data, look_back)

# Reshape data for LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=10)

# Make predictions on the test data
train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)

# Inverse transform the predictions and actual values
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Plot training and test predictions
st.subheader("Train and Test Predictions")
train = data[:train_size]
test = data[train_size + look_back:]
test = test.reset_index(drop=True)  # Align index with test predictions
test['Predictions'] = test_predictions

fig2, ax2 = plt.subplots()
ax2.plot(train['Date'], train['Close'], label='Train Data')
ax2.plot(test['Date'], test['Close'], label='Actual Price')
ax2.plot(test['Date'], test['Predictions'], label='Predicted Price')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price in USD')
ax2.legend()
st.pyplot(fig2)

# Calculate RMSE and MSE for model accuracy
rmse = np.sqrt(mean_squared_error(y_test[0], test_predictions[:, 0]))
mse = mean_squared_error(y_test[0], test_predictions[:, 0])
accuracy_percentage = (1 - rmse / np.mean(y_test[0])) * 100

st.write("Model Accuracy Metrics:")
st.write(f"RMSE: {rmse}")
st.write(f"MSE: {mse}")
st.write(f"Accuracy: {accuracy_percentage:.2f}%")

# Future Prediction for user-defined days (1-30 days)
st.subheader("Future Bitcoin Price Prediction")
days_to_predict = st.slider("Select the number of days for prediction (1-30):", 1, 30)

# Prepare the last 60 days of data for prediction
last_60_days = scaled_data[-look_back:]
x_future = last_60_days.reshape((1, look_back, 1))

# Predict future prices
future_predictions = []
for _ in range(days_to_predict):
    pred_price = model.predict(x_future)
    future_predictions.append(pred_price[0, 0])
    reshaped_pred=np.reshape(pred_price, (1, 1, 1))
    x_future = np.append(x_future[:, 1:, :], reshaped_pred, axis=1)

# Inverse transform predictions and display in a table
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, days_to_predict + 1)]

# Display future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})
st.write(future_df)

# Plot future predictions
fig3, ax3 = plt.subplots()
ax3.plot(data['Date'], data['Close'], label='Historical Price')
ax3.plot(future_df['Date'], future_df['Predicted Price'], label='Future Predictions')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price in USD')
ax3.legend()
st.pyplot(fig3)