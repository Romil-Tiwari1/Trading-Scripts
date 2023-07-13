# Meta Inc. Stock Price Prediction
This Python script predicts the future closing prices of Meta Inc.'s stock using Linear Regression and ARIMA models.

# Dependencies
To run this script, you need the following Python packages:

pandas
numpy
sklearn
statsmodels
matplotlib

# Usage
Update the data_path variable with the path to your CSV file containing historical stock prices for Meta Inc. The CSV file should contain at least the following columns: 'Date', 'Close/Last'. 'Date' should be in a format that pandas can convert to datetime and 'Close/Last' should contain the closing prices with a dollar sign.

Update the latest_close_price variable with the most recent closing price.

The script performs the following steps:

1. Loads the data from the specified CSV file.
2. Converts 'Date' to datetime format.
3. Converts 'Close/Last' to float after removing the dollar sign.
4. Transforms 'Date' into the number of days since the start of the dataset.
5. Fits a Linear Regression model and an ARIMA model to the historical data.
6. Predicts the closing prices for the next 7 days using both models.
7. Updates the models with the latest closing price.
8. Plots the historical data and the predicted prices from both models.
9. The script generates a plot showing the historical closing prices of Meta Inc.'s stock, along with the predicted prices from the Linear Regression and ARIMA models for the next 7 days.

# Learning Outcomes
By working with this script, you can gain an understanding of the following concepts:

1. Data Preprocessing: The script demonstrates how to prepare your data for analysis. This includes converting data types, handling dates, and transforming data into a format suitable for modeling.

2. Linear Regression: This script provides a practical application of Linear Regression, which is a fundamental algorithm in Machine Learning. You can see how to fit a model to your data and use it for future predictions.

3. Time Series Forecasting with ARIMA: ARIMA is a popular method for forecasting in time-series data. This script shows how to fit an ARIMA model and use it to make predictions.

4. Model Update: The script demonstrates how to update your models with new data. This is a common scenario in real-world applications where new data becomes available over time.

5. Data Visualization: The script uses Matplotlib, a powerful library for creating static, animated, and interactive visualizations in Python. You'll see how to create line plots, add labels, title, and legend, and customize your plot.

# Implemented Financial Concepts
Stock Prices: This script works with historical stock price data, providing hands-on experience with a key concept in finance.

1. Predictive Modeling: The script applies machine learning models to predict future stock prices. This is a common task in financial forecasting.

2. Time-Series Analysis: Financial data like stock prices is typically a time-series. This script demonstrates how to work with time-series data, transform it for modeling, and perform time-series forecasting.

3. Moving Average: The ARIMA model implemented in this script includes the concept of a moving average, a commonly used technique in financial analysis.

# Notes
This script is for demonstration purposes only and should not be used for financial advice. The accuracy of the models' predictions is dependent on the quality and quantity of the historical data provided.