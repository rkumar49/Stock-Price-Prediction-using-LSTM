import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import datetime

st.set_page_config(page_title="Stock Trend Prediction", layout="centered")

st.title("📈 Stock Trend Prediction Using LSTM")
st.markdown("Enter a stock ticker below and select a date range to view predictions.")

# Sidebar inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL")
years = st.slider("Select number of years", 1, 15, 10)

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365 * years)

# Fetch stock data
df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']].dropna()

# Show statistics
st.subheader(f"📊 Descriptive Statistics for {ticker.upper()} (Last {years} Years)")
st.write(df.describe())

# Scalar calculations for storytelling
min_price = round(float(df['Close'].min()), 2)
max_price = round(float(df['Close'].max()), 2)
mean_price = round(float(df['Close'].mean()), 2)
std_dev = round(float(df['Close'].std()), 2)

# Volatility classification
if std_dev > 0.2 * mean_price:
    volatility = "high 🔥"
elif std_dev > 0.1 * mean_price:
    volatility = "moderate ⚠️"
else:
    volatility = "low ✅"

# Display explanation
st.info(
    f"""
    - 📉 **Min Price:** ${min_price}  
    - 📈 **Max Price:** ${max_price}  
    - 🧮 **Average Price:** ${mean_price}  
    - 📊 **Volatility Level:** {volatility} (std dev = {std_dev})
    """
)

# Plot raw closing price
st.subheader("📈 Closing Price Over Time")
fig1, ax1 = plt.subplots()
ax1.plot(df['Close'], label='Close Price', color='blue')
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

# Moving averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

st.subheader("📉 Closing Price with MA100 and MA200")
fig2, ax2 = plt.subplots()
ax2.plot(df['Close'], label='Close Price')
ax2.plot(ma100, 'r', label='MA100')
ax2.plot(ma200, 'g', label='MA200')
ax2.legend()
st.pyplot(fig2)

# Trend interpretation
latest = float(df['Close'].iloc[-1])
ma100_last = float(ma100.iloc[-1])
ma200_last = float(ma200.iloc[-1])

if latest > ma100_last > ma200_last:
    trend = "⬆️ Bullish"
elif latest < ma100_last < ma200_last:
    trend = "⬇️ Bearish"
else:
    trend = "⚖️ Sideways/Unclear"

st.success(
    f"""
    **Market Trend Analysis**  
    - Latest Close Price: **${latest:.2f}**  
    - 100-day MA: **${ma100_last:.2f}**  
    - 200-day MA: **${ma200_last:.2f}**  
    - **Trend Direction:** {trend}
    """
)

# Prepare data for prediction
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load model
model = load_model('keras_model.h5')

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_predicted = model.predict(x_test)

# Inverse scaling
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Prediction chart
st.subheader("🤖 Predicted Price vs Actual Price")
fig3, ax3 = plt.subplots()
ax3.plot(y_test, 'b', label='Actual Price')
ax3.plot(y_predicted, 'r', label='Predicted Price')
ax3.set_xlabel("Time")
ax3.set_ylabel("Price")
ax3.legend()
st.pyplot(fig3)

# Model performance
rmse = mean_squared_error(y_test, y_predicted, squared=False)

if rmse < 5:
    performance = "Excellent 🔥"
elif rmse < 15:
    performance = "Good 👍"
else:
    performance = "Needs Improvement ⚠️"

st.warning(
    f"""
    **Model Performance**  
    - RMSE (Root Mean Squared Error): **{rmse:.2f}**  
    - Interpretation: **{performance}**
    """
)
