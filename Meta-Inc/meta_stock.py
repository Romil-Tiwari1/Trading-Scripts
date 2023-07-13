import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the data
data_path = "/Users/romiltiwari/Documents/HistoricalData_1689206884387.csv"
data = pd.read_csv(data_path)

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Remove dollar sign and convert 'Close/Last' to float
data['Close/Last'] = data['Close/Last'].str.replace('$', '').astype(float)

# Sort the data in ascending order of date
data = data.sort_values('Date').reset_index(drop=True)

# Transform 'Date' into number of days since the start
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Define independent variable (Days) and dependent variable (Close/Last)
X = data['Days'].values.reshape(-1,1)
y = data['Close/Last'].values

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict closing prices for the next 7 days
days_since_start = X.max() + np.arange(1, 8).reshape(-1,1) # days since the start for the next 7 days
predicted_close_prices = model.predict(days_since_start)

# Fit the ARIMA model
y_diff = np.diff(y)
model_arima = ARIMA(y_diff, order=(5,1,0))
model_arima_fit = model_arima.fit()

# Predict the difference of closing prices for the next 7 days
predicted_diff = model_arima_fit.predict(start=len(y_diff), end=len(y_diff) + 6)

# The predictions are in terms of the first difference, so we need to add the last known price to get the predicted price
predicted_close_prices_arima = np.cumsum(np.concatenate([np.array([y[-1]]), predicted_diff]))[1:]

# Update the models with the latest closing price
latest_close_price = 309.49  # Replace this with the latest closing price
data = data.append({'Date': pd.to_datetime('2023-07-12'), 'Close/Last': latest_close_price, 'Days': data['Days'].max() + 1}, ignore_index=True)

# Re-define the independent variable (Days) and dependent variable (Close/Last)
X_updated = data['Days'].values.reshape(-1,1)
y_updated = data['Close/Last'].values

# Fit the model to the updated data
model.fit(X_updated, y_updated)

# Predict closing prices for the next 7 days
days_since_start_updated = X_updated.max() + np.arange(1, 8).reshape(-1,1) # days since the start for the next 7 days
predicted_close_prices_updated = model.predict(days_since_start_updated)

# Update the ARIMA model
y_diff_updated = np.diff(y_updated)
model_arima_updated = ARIMA(y_diff_updated, order=(5,1,0))
model_arima_fit_updated = model_arima_updated.fit()

# Predict the difference of closing prices for the next 7 days
predicted_diff_updated = model_arima_fit_updated.predict(start=len(y_diff_updated), end=len(y_diff_updated) + 6)

# The predictions are in terms of the first difference, so we need to add the last known price to get the predicted price
predicted_close_prices_arima_updated = np.cumsum(np.concatenate([np.array([y_updated[-1]]), predicted_diff_updated]))[1:]

# Define the dates for the next 7 days
dates = pd.date_range(data['Date'].max() + pd.Timedelta(days=1), periods=7)

# Plot the historical data
plt.figure(figsize=(14,7))
plt.plot(data['Date'], data['Close/Last'], label='Historical Close Price')

# Plot the predictions from the linear regression model
plt.plot(dates, predicted_close_prices_updated, label='Updated Linear Regression Predictions')

# Plot the predictions from the ARIMA model
plt.plot(dates, predicted_close_prices_arima_updated, label='Updated ARIMA Predictions')

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Updated Meta Inc. Stock Price Predictions')
plt.legend()
plt.grid(True)
plt.show()