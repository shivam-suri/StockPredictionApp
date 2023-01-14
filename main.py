import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.set_page_config(page_title="Stock Price Prediction App",
                   page_icon="📈")

st.title("Stock Price Prediction App")

# Download Data

ticker = st.text_input("Enter Ticker Symbol:")
start_year = st.slider("Select Model Prediction Starting Date:", 2012, 2022)

df = yf.download(ticker)
public_year = df.reset_index()["Date"][1].year

if public_year >= start_year:
    start_year = public_year + 1

start = dt.datetime(2011, 1, 1)
end = dt.datetime(start_year + 1, 1, 1)

df = yf.download(ticker, start, end)

# Prepare Data

scaler = MinMaxScaler(feature_range=(0, 1))

prediction_days = 100

# Load Test Data

model = load_model("StockPrediction_Model")

test_start = dt.datetime(start_year, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(ticker, test_start, test_end)
original_prices = test_data["Close"].values

total_data = pd.concat((df["Close"], test_data["Close"]), axis=0)

model_inputs = total_data[len(total_data) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

# Make Predictions on Test Data

x_test = []

for i in range(prediction_days, model_inputs.shape[0]):
    x_test.append(model_inputs[i-prediction_days:i, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Predict Next Day

real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

# Show Graph

st.subheader("Accuracy of the prediction:")
stock_chart = plt.figure(figsize=(12, 6))
plt.plot(original_prices, color="black", label=f"Actual {ticker} price")
plt.plot(predicted_prices, color="red", label=f"Predicted {ticker} price")
plt.title(f"{ticker} Share Price")
plt.xlabel("Time (Days)")
plt.ylabel(f"{ticker} Share Price")
plt.legend()
st.pyplot(stock_chart)

# Show Prediction Value

prediction = prediction[0][0]
prediction = format(prediction, ".2f")
st.subheader(f"Next trading day predicted value for {ticker}: ${prediction}")

st.subheader("")
st.subheader("")

# DISCLAIMER

st.subheader("Message from the creator:")
st.write("Thanks for using my Stock Prediction App!")
st.write("This app will work for all equities listed on Yahoo Finance, including:")
st.write("- Stocks")
st.write("- ETFs")
st.write("- Cryptos")
st.write("Before you begin, please see the following:")
st.write("- Ensure the ticker symbol is correctly spelled, with all characters included")
st.write("- The earlier you choose for the starting date, the more accurate the results")
st.write("This app is made for entertainment purposes and is not a financial advisor")
st.write("Please ensure you are doing your due diligence before investing!")