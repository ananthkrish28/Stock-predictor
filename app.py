import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# ----------------- Streamlit Setup --------------------
st.title('Stock Trend Prediction with LSTM, RF & ARIMA')

start = '2010-01-01'
end = '2025-12-31'

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2010 - 2025')
st.write(df.describe())

# ----------------- Visualization --------------------
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

# MA 100
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

# MA 100 & 200
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df['Close'], 'b')
st.pyplot(fig)

# ----------------- RSI Calculation --------------------
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

df = compute_rsi(df)
st.subheader('RSI Chart')
fig = plt.figure(figsize=(12, 4))
plt.plot(df['RSI'], label='RSI')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
st.pyplot(fig)

# ----------------- LSTM Model --------------------
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model.keras')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Prediction
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted *= scale_factor
y_test = np.array(y_test) * scale_factor

# Plot
st.subheader('LSTM Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='LSTM Predicted Price')
plt.legend()
st.pyplot(fig2)

# ----------------- Random Forest --------------------
st.subheader('Random Forest Model Prediction')
df_rf = df.dropna()
X_rf = df_rf[['Open', 'High', 'Low', 'Volume', 'RSI']]
y_rf = df_rf['Close']
split = int(len(df_rf) * 0.8)
X_train_rf, X_test_rf = X_rf[:split], X_rf[split:]
y_train_rf, y_test_rf = y_rf[:split], y_rf[split:]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_rf, y_train_rf)
y_pred_rf = rf.predict(X_test_rf)

rf_rmse = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
st.write(f"Random Forest RMSE: {rf_rmse:.2f}")

fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test_rf.values, label='Original Price')
plt.plot(y_pred_rf, label='RF Predicted Price', color='orange')
plt.legend()
st.pyplot(fig3)

# ----------------- ARIMA --------------------

# ARIMA section
st.subheader("ARIMA Forecast (From 2010 to 2025)")

# Ensure index is datetime
df.index = pd.to_datetime(df.index)

# Ensure 'Close' prices are used and clean
close_prices = df['Close'].dropna()

# Train ARIMA model
from statsmodels.tsa.arima.model import ARIMA
model_arima = ARIMA(close_prices, order=(5, 1, 0))
model_arima_fit = model_arima.fit()

# Forecast next 30 days
forecast_steps = 30
forecast = model_arima_fit.forecast(steps=forecast_steps)

# Generate future date range
last_date = close_prices.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)

# Remove forecasts that fall beyond 2025
future_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
future_df = future_df[future_df['Date'].dt.year <= 2025]

# Plot actual + forecast
fig4 = plt.figure(figsize=(12, 6))
plt.plot(close_prices, label='Actual Prices (2010–2025)', color='blue')
plt.plot(future_df['Date'], future_df['Forecast'], label='ARIMA Forecast (Next 30 Days)', color='purple')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("ARIMA Forecast with Full Historical Data")
plt.legend()
st.pyplot(fig4)
