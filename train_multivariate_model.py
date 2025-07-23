import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import datetime

# Download stock data (for training)
ticker = "AAPL"
end = datetime.date.today()
start = end - datetime.timedelta(days=365 * 10)

df = yf.download(ticker, start=start, end=end)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Split into training/testing
data_training = df[0:int(len(df)*0.70)]
data_testing = df[int(len(df)*0.70):]

# Scale
scaler = MinMaxScaler()
scaled_training = scaler.fit_transform(data_training)

# Prepare sequences
x_train = []
y_train = []

for i in range(100, scaled_training.shape[0]):
    x_train.append(scaled_training[i-100:i])
    y_train.append(scaled_training[i, 3])  # 'Close' is at index 3

x_train, y_train = np.array(x_train), np.array(y_train)

# Build LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(100, activation='relu', return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save("keras_model_multivariate.h5")
print("✅ Model trained and saved as keras_model_multivariate.h5")
