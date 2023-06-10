import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the stock price data
data = pd.read_csv('stock_prices.csv')

# Convert the date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set the date as the index
data.set_index('Date', inplace=True)

# Plot the stock prices
data['Close'].plot(figsize=(10, 6))
plt.title('Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Split the data into training and testing sets
train_data = data['Close'][:int(0.8 * len(data))]
test_data = data['Close'][int(0.8 * len(data)):]

# Create and train the ARIMA model
model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, typ='levels')

# Visualize the predictions
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
