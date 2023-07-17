import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Defining the functions used in script
def create_xy_datasets(data):
    x, y = [], []
    for i in range(60, len(data)):
        x.append(data[i-60:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def build_model(input_data):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_data.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    return model

def print_errors(mae, mse, rmse, price_type):
    print(f'\n{price_type.upper()} PRICE ERRORS:')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')

def plot_data(train_data, valid_data, y_label):
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.plot(train_data)
    plt.plot(valid_data[['Predictions']])
    plt.legend(['Train', 'Predictions'], loc='lower right')
    plt.show()

yf.pdr_override()

# User input for stock symbol
symbol = input("Please enter the stock symbol: ")

# Dates
end_date = pd.to_datetime('today')
start_date = end_date - pd.DateOffset(years=5)

# Get data
df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

# Extract 'Close' and 'Open' price columns
data_close = df.filter(['Close'])
data_open = df.filter(['Open'])

# Convert to numpy array
dataset_close = data_close.values
dataset_open = data_open.values

# Calculate lengths of datasets
training_data_len_close = math.ceil(len(dataset_close) * .8)
training_data_len_open = math.ceil(len(dataset_open) * .8)

# Scale data
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaled_data_close = scaler_close.fit_transform(dataset_close)
scaler_open = MinMaxScaler(feature_range=(0, 1))
scaled_data_open = scaler_open.fit_transform(dataset_open)

# Training data set creation
train_data_close = scaled_data_close[0:int(training_data_len_close), :]
train_data_open = scaled_data_open[0:int(training_data_len_open), :]

# x_train and y_train data sets
x_train_close, y_train_close = create_xy_datasets(train_data_close)
x_train_open, y_train_open = create_xy_datasets(train_data_open)

# Reshape data to 3D for LSTM
x_train_close = np.reshape(x_train_close, (x_train_close.shape[0], x_train_close.shape[1], 1))
x_train_open = np.reshape(x_train_open, (x_train_open.shape[0], x_train_open.shape[1], 1))

# LSTM model
model_close = build_model(x_train_close)
model_open = build_model(x_train_open)

# Model compilation
model_close.compile(optimizer='adam', loss='mean_squared_error')
model_open.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping definition
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# Model training
model_close.fit(x_train_close, y_train_close, batch_size=1, epochs=100, callbacks=[early_stopping], validation_split=0.2)
model_open.fit(x_train_open, y_train_open, batch_size=1, epochs=100, callbacks=[early_stopping], validation_split=0.2)

# Testing data set creation
test_data_close = scaled_data_close[training_data_len_close - 60:, :]
test_data_open = scaled_data_open[training_data_len_open - 60:, :]

# x_test and y_test data sets creation
x_test_close, y_test_close = create_xy_datasets(test_data_close)
x_test_open, y_test_open = create_xy_datasets(test_data_open)

# Data reshaping
x_test_close = np.reshape(x_test_close, (x_test_close.shape[0], x_test_close.shape[1], 1))
x_test_open = np.reshape(x_test_open, (x_test_open.shape[0], x_test_open.shape[1], 1))

# Model prediction
predictions_close = model_close.predict(x_test_close)
predictions_open = model_open.predict(x_test_open)

# Inverse transformation
predictions_close = scaler_close.inverse_transform(predictions_close)
predictions_open = scaler_open.inverse_transform(predictions_open)

# Error metrics
mae_close = mean_absolute_error(predictions_close, y_test_close)
mse_close = mean_squared_error(predictions_close, y_test_close)
rmse_close = math.sqrt(mean_squared_error(predictions_close, y_test_close))

mae_open = mean_absolute_error(predictions_open, y_test_open)
mse_open = mean_squared_error(predictions_open, y_test_open)
rmse_open = math.sqrt(mean_squared_error(predictions_open, y_test_open))

# Print errors
print_errors(mae_close, mse_close, rmse_close, "close")
print_errors(mae_open, mse_open, rmse_open, "open")

# Data for plot
train_close = data_close[:training_data_len_close]
valid_close = data_close[training_data_len_close:]
valid_close['Predictions'] = predictions_close

train_open = data_open[:training_data_len_open]
valid_open = data_open[training_data_len_open:]
valid_open['Predictions'] = predictions_open

# Plot data
plot_data(train_close, valid_close, "Close Price USD ($)")
plot_data(train_open, valid_open, "Open Price USD ($)")

# Print predicted values
print("Predicted close prices:")
print(valid_close)
print("\nPredicted open prices:")
print(valid_open)

# Table creation
close_table = pd.DataFrame({"Actual Close": y_test_close.flatten(), "Predicted Close": predictions_close.flatten()})
open_table = pd.DataFrame({"Actual Open": y_test_open.flatten(), "Predicted Open": predictions_open.flatten()})

# Display tables
print("\nClose Prices Table")
display(close_table)
print("\nOpen Prices Table")
display(open_table)
