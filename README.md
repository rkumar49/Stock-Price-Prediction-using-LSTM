# 📈 Stock Trend Prediction Using LSTM (Univariate & Multivariate)

This project aims to analyze and predict stock prices using **Long Short-Term Memory (LSTM)** deep learning models, built in Python and deployed via **Streamlit**.  
The app allows users to explore historical stock data, understand feature relationships, and view prediction performance.

---

## 🧠 Project Goals

- ✅ Perform **univariate time series prediction** using only the `Close` price
- ✅ Develop a **multivariate LSTM model** using features: `Open`, `High`, `Low`, `Close`, and `Volume`
- ✅ Build a **Streamlit web application** to visualize the trends, predictions, and model performance
- ✅ Deploy the application on **AWS SageMaker** using a model saved in **S3**
- 🔮 Extend into **future forecasting** using recursive LSTM techniques

---

## 📦 Technologies Used

| Tool/Library        | Purpose                          |
|---------------------|----------------------------------|
| `Python`            | Core programming                 |
| `yfinance`          | Real-time stock data             |
| `pandas`, `numpy`   | Data wrangling                   |
| `matplotlib`, `seaborn` | Visualization              |
| `scikit-learn`      | Data scaling, RMSE calculation   |
| `Keras / TensorFlow`| LSTM deep learning models        |
| `Streamlit`         | Web UI for interactive dashboard |
| `AWS SageMaker`     | Cloud deployment                 |
| `Amazon S3`         | Model file storage               |

---

## 1️⃣ Univariate Time Series Analysis

### 📌 Input Feature:
- `Close` price only

### 📉 Workflow:
1. Fetch stock data from Yahoo Finance
2. Preprocess using `MinMaxScaler`
3. Train LSTM on historical window (e.g., 100 previous days)
4. Predict on test split (30% of data)
5. Evaluate with RMSE, visualize predicted vs actual

### 📊 Output:
- Closing price prediction over past 10 years
- Model performance based on root mean square error (RMSE)
- Trend interpretation using Moving Averages

---

## 2️⃣ Multivariate Time Series Prediction (LSTM)

### 📌 Input Features:
- `Open`, `High`, `Low`, `Close`, `Volume`

### 🔍 Visualizations:
- Descriptive statistics & volatility rating
- Line chart of features
- Correlation heatmap
- Moving Averages (MA100, MA200)
- Predicted vs Actual chart using trained LSTM

### 🧠 Model:
- Trained LSTM model saved as `keras_model_multivariate.h5`
- Takes 100-day windows of all features as input
- Predicts next-day `Close` price

### 📏 Evaluation:
- RMSE used for accuracy
- Prediction quality ranked (Excellent, Moderate, Poor)
- Trend indicator (Bullish, Bearish, Neutral)

---

## 3️⃣ Forecasting (To Predict Future Days)

While current models predict known test data, we implement the concept of **forecasting future prices** (e.g., next 30 days):

### 🔄 Approach:
- Use recursive LSTM: predict one day, append it, re-predict
- Continue N times into future (without known features)

### 🧪 Forecast Mode:
- (In separate branch or planned extension)
- Ideal for real-world stock prediction dashboards

---

## 4️⃣ Streamlit Web Application

A real-time interactive dashboard built using Streamlit:
- Users can select:
  - Stock ticker
  - Date range (1–15 years)
- Application dynamically updates:
  - Charts
  - Correlation map
  - Model predictions
  - Trend summaries
  - Descriptive storytelling

### 🔧 Run Locally

```bash
git clone https://github.com/yourname/stock-lstm-app.git
cd stock-lstm-app
pip install -r requirements.txt
streamlit run app.py
