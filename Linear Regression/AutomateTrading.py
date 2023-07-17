import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

yf.pdr_override()

# Take the stock symbol as an input from the user
symbol = input("Please enter the stock symbol: ")

# We'll use data from the past 5 years
end_date = pd.to_datetime('today')
start_date = end_date - pd.DateOffset(years=5)

# Fetch the data
df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

# Add a column for the 14-day moving average
df['MA14'] = df['Close'].rolling(window=14).mean()

# Add a column for the standard deviation of the returns (our volatility measure)
df['Volatility'] = df['Close'].pct_change().rolling(window=14).std()

# Drop any rows with missing values
df.dropna(inplace=True)

# Define our inputs for the model
inputs = df[['Close', 'Volume', 'MA14', 'Volatility']]

# Define our output for closing price prediction
output_close = df['Close']

# Split the data into training and testing sets for closing price prediction (80/20 split)
X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(inputs, output_close, test_size=0.2, shuffle=False)

# Build the Linear Regression model for closing price prediction
lr_model_close = LinearRegression()
lr_model_close.fit(X_train_close, y_train_close)

# Test the model for closing price prediction
lr_predictions_close = lr_model_close.predict(X_test_close)
lr_mse_close = mean_squared_error(y_test_close, lr_predictions_close)
print(f'Linear Regression MSE for Closing Price: {round(lr_mse_close, 2)}')

# Predict the next day's closing price
next_day_inputs = df[['Close', 'Volume', 'MA14', 'Volatility']].values[-1].reshape(1, -1)
lr_pred_close = lr_model_close.predict(next_day_inputs)

# Define our output for opening price prediction
output_open = df['Open']

# Split the data into training and testing sets for opening price prediction (80/20 split)
X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(inputs, output_open, test_size=0.2, shuffle=False)

# Build the Linear Regression model for predicting opening prices
lr_model_open = LinearRegression()
lr_model_open.fit(X_train_open, y_train_open)

# Test the model for opening price prediction
lr_predictions_open = lr_model_open.predict(X_test_open)
lr_mse_open = mean_squared_error(y_test_open, lr_predictions_open)
print(f'Linear Regression MSE for Opening Price: {round(lr_mse_open, 2)}')

# Predict the next day's opening price
lr_pred_open = lr_model_open.predict(next_day_inputs)

# Calculate and display the percentage change between the last day's opening price and the predicted next day's opening price
percentage_change_open = ((lr_pred_open - df["Open"].values[-1]) / df["Open"].values[-1]) * 100
percentage_change_sign_open = '+' if percentage_change_open >= 0 else '-'

print(f'Previous Day\'s Opening Price: {round(df["Open"].values[-1], 2)}')
print(f'Percentage Change Between Previous Day Open Price and Predicted Next Day Open Price: {percentage_change_sign_open}{round(abs(percentage_change_open[0]), 2)}%')

# Calculate and display the percentage change between the last day's closing price and the predicted next day's opening price
percentage_change_close_open = ((lr_pred_open - df["Close"].values[-1]) / df["Close"].values[-1]) * 100
percentage_change_sign_close_open = '+' if percentage_change_close_open >= 0 else '-'

print(f'Last data considered for prediction on {df.index[-1].date()} with closing price {round(df["Close"].values[-1], 2)}')
print(f'Percentage Change Between Previous Day\'s Close Price and Predicted Next Day\'s Open Price: {percentage_change_sign_close_open}{round(abs(percentage_change_close_open[0]), 2)}%')

print(f'Linear Regression Prediction for the next day\'s opening price: {round(lr_pred_open[0], 2)}')
