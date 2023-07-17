# Automated Stock Trading System

This project is an automated stock trading system that uses linear regression to predict the next day's opening and closing prices for a given stock.

## Features

- User can specify the stock symbol to analyze.
- Uses the past 5 years of daily stock data from Yahoo Finance API.
- Calculates additional features: 14-day moving average and the standard deviation of the returns (volatility).
- Trains and evaluates a linear regression model for predicting the next day's opening and closing prices.
- Reports the previous day's opening and closing prices, the predicted prices, and the percentage changes.

## How to Run

1. Make sure you have the required dependencies installed: pandas, pandas_datareader, yfinance, sklearn.
2. Run the script in a Python environment and enter the stock symbol when prompted.

## Future Improvements

- Incorporate more features for better prediction accuracy.
- Implement other prediction models such as ARIMA, LSTM.
- Evaluate models using additional metrics such as R-squared, MAE.
- Automate the stock trading based on the predictions.

## Lessons Learned

1. **Collecting stock data using the Yahoo Finance API**: The script uses the `yfinance` library, which allows us to easily fetch historical market data from Yahoo Finance. We learned how to fetch and structure the data in a way that is suitable for further analysis and modeling.

2. **Feature Engineering for Stock Price Prediction**: In addition to the base features provided by the Yahoo Finance data, we created additional features to help improve our model's predictive power. These included a 14-day moving average, which smooths out price fluctuations and highlights the underlying trend, and a volatility measure, which quantifies the degree of variation in the stock's trading price over a certain period. These features provide more context to the model about the stock's past behavior.

3. **Training and Evaluating a Linear Regression Model for Stock Price Prediction**: We learned how to use the `sklearn` library to create a Linear Regression model and fit it to our data. We split our data into training and testing sets, and evaluated the model's performance by calculating the mean squared error (MSE) on the testing set. The MSE provides a measure of how well the model is able to predict the stock's opening and closing prices.

4. **Predicting Future Stock Prices Using the Trained Model**: After training the model, we used it to predict the next day's opening and closing prices. This involves feeding the most recent data into the model and extracting the output, which is the predicted price. This process showed us how the model can be used in a practical setting to anticipate stock price movements.

5. **Handling Real-World Data**: One key takeaway from this project is understanding and handling the uncertainties and irregularities of real-world data. For instance, we learned the importance of handling missing values (NaNs) in our dataset, which can often occur in financial data.

6. **Python Programming and Libraries**: The project also provided hands-on experience with Python programming, specifically for data manipulation (using `pandas`) and data visualization. It further reinforced the understanding of using Python libraries for data analysis and machine learning.

