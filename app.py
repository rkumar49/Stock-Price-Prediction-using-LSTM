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

st.set_page_config(page_title="Multivariate Stock LSTM", layout="centered")
st.title("📈 Multivariate Stock Trend Prediction Using LSTM")

# Input
ticker = st.text_input("Enter Stock Ticker", "AAPL")
years = st.slider("Select number of years", 1, 15, 10)
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365 * years)

# Data fetch
df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

st.subheader(f"📊 Feature Overview for {ticker.upper()}")
st.dataframe(df.tail())

# Descriptive Statistics Table
st.subheader("📋 Descriptive Statistics")
st.dataframe(df.describe())

# Basic statistics for insights
min_price = round(float(df['Close'].min()), 2)
max_price = round(float(df['Close'].max()), 2)
mean_price = round(float(df['Close'].mean()), 2)
std_dev = round(float(df['Close'].std()), 2)

if std_dev > 0.2 * mean_price:
    volatility = "High 🔥"
elif std_dev > 0.1 * mean_price:
    volatility = "Moderate ⚠️"
else:
    volatility = "Low ✅"

st.markdown("### 📘 Insight from Statistics")
st.info(f"""
- **Average Closing Price:** ${mean_price}
- **Price Range:** ${min_price} to ${max_price}
- **Volatility:** {volatility} (Std Dev: {std_dev})
- Stocks with high volatility can offer higher returns but also higher risk.
""")

# Line chart for all features
st.subheader("📈 Feature Time Series")
fig1, ax1 = plt.subplots(figsize=(12, 5))
for col in df.columns:
    ax1.plot(df[col], label=col)
ax1.set_ylabel("Price / Volume")
ax1.legend()
st.pyplot(fig1)

st.markdown("### 📘 Insight on Price Movement")
st.success(f"""
- **{ticker.upper()}** shows historical price and volume trends.
- Sudden **Volume spikes** can signal events or institutional activity.
- Divergence between volume and price may signal reversals.
""")

# Correlation heatmap
st.subheader("📊 Feature Correlation Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.markdown("### 📘 Insight on Feature Relationships")
st.info("""
- **Strong correlations**:
    - `Close`, `Open`, `High`, and `Low` have strong positive correlation (typically >0.9).
    - This makes them **valuable features for price prediction**.
- **Weak correlations**:
    - `Volume` has low or even negative correlation with prices.
    - Suggests volume behaves **independently** — useful for identifying unusual activity, not trend.
""")

# Multivariate Preprocessing
data_training = df[0:int(len(df)*0.70)]
data_testing = df[int(len(df)*0.70):]

scaler = MinMaxScaler()
scaled_training = scaler.fit_transform(data_training)

x_train, y_train = [], []
for i in range(100, scaled_training.shape[0]):
    x_train.append(scaled_training[i-100:i])
    y_train.append(scaled_training[i, 3])  # 3 = Close

x_train, y_train = np.array(x_train), np.array(y_train)

# Load model
model = load_model('keras_model_multivariate.h5')

# Prepare test set
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 3])  # 3 = Close

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[3]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Prediction date range
prediction_start = data_testing.index[0].strftime('%Y-%m-%d')
prediction_end = data_testing.index[-1].strftime('%Y-%m-%d')

st.markdown(f"### 🗓️ Prediction Period: {prediction_start} to {prediction_end}")

# Plot prediction
st.subheader("🧠 Predicted vs Actual Closing Prices")
fig3, ax3 = plt.subplots()
ax3.plot(y_test, label="Actual Price", color='blue')
ax3.plot(y_predicted, label="Predicted Price", color='red')
ax3.set_ylabel("Price")
ax3.set_xlabel("Time Steps")
ax3.legend()
st.pyplot(fig3)

# Evaluation
rmse = mean_squared_error(y_test, y_predicted, squared=False)

if rmse < 5:
    model_comment = "🟢 Excellent fit — predictions follow trends very well."
elif rmse < 15:
    model_comment = "🟡 Moderate accuracy — trend is captured, some deviation."
else:
    model_comment = "🔴 High error — model may need more training or better features."

st.markdown("### 📘 Model Performance Insight")
st.warning(f"""
- **RMSE:** {rmse:.2f}
- {model_comment}
""")

# Prediction vs actual sample
comparison_df = pd.DataFrame({
    'Date': data_testing.index[:len(y_test)],
    'Actual Price': y_test.flatten(),
    'Predicted Price': y_predicted.flatten()
}).set_index('Date')

st.subheader("📋 Sample Prediction vs Actual")
st.dataframe(comparison_df.tail(10))

# Trend Summary
latest = float(df['Close'].iloc[-1])
ma100_last = float(df['Close'].rolling(100).mean().iloc[-1])
ma200_last = float(df['Close'].rolling(200).mean().iloc[-1])

if latest > ma100_last > ma200_last:
    trend = "⬆️ Bullish"
elif latest < ma100_last < ma200_last:
    trend = "⬇️ Bearish"
else:
    trend = "⚖️ Sideways/Neutral"

st.subheader("📌 Summary of Current Trend")
st.success(f"""
- **Latest Close Price:** ${latest:.2f}
- **MA100:** ${ma100_last:.2f}, **MA200:** ${ma200_last:.2f}
- **Trend Indicator:** {trend}
""")

# Final Summary Block
st.markdown("## 🧾 Final Report")
st.info(f"""
- 📈 **Stock Ticker:** {ticker.upper()}
- 📅 **Date Range:** {start_date} to {end_date}
- 🧠 **Model RMSE:** {rmse:.2f}
- 🗓️ **Prediction Period:** {prediction_start} to {prediction_end}
- 🔁 **Trend Analysis:** {trend}
- 📊 **Volatility Level:** {volatility}
- ✅ Multivariate model using: Open, High, Low, Close, Volume.
""")
