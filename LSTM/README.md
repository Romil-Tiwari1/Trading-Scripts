# Stock Price Predictor Using LSTM
This project involves building a stock price prediction model using Long Short-Term Memory (LSTM), a type of Recurrent Neural Network (RNN). It collects historical stock data, preprocesses it, trains an LSTM model on it, makes predictions, evaluates the model's performance, and visualizes the results.

## Requirements
pandas
numpy
pandas_datareader
yfinance
sklearn
tensorflow
matplotlib
seaborn

## Data Collection
The project uses Yahoo Finance API to collect historical stock price data. The user can input the stock symbol of their choice, and the system will fetch the data for the past five years.

## Preprocessing
The collected data undergoes several preprocessing steps:

1. Extraction of 'Open' and 'Close' prices.
2. Conversion to numpy arrays.
3. Scaling the data using MinMaxScaler.
4. Creation of training and testing datasets.
5. Model Building
6. The LSTM model is built using TensorFlow. The model architecture includes two LSTM layers and two Dense layers.

## Training
The LSTM model is trained on the dataset. Early stopping is also implemented to prevent overfitting. The model is trained for a maximum of 100 epochs, with a batch size of 1.

## Testing & Prediction
The trained model is used to predict 'Open' and 'Close' prices on the testing dataset.

## Error Calculation
The model's performance is evaluated by calculating the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for the predictions.

## Visualization
The actual and predicted prices are plotted for visual comparison. Two separate plots are generated for 'Open' and 'Close' prices.

## Tabulation
The actual and predicted prices are also displayed in tabular form for an easy comparison.

## Future Improvements
The project can be improved in several ways:

1. Incorporating more features into the model, like volume, high, low prices, and other technical indicators.
2. Optimizing the model architecture and hyperparameters.
3. Using ensemble methods or more sophisticated models.
4. Including a larger dataset for training.
5. Applying more advanced techniques for handling overfitting, such as dropout or regularization.

## Lessons Learned

This project was a great opportunity to learn and refine several skills related to data science, machine learning, and financial analysis. Here are some of the key lessons learned:

1. **Data Collection and Preprocessing**:
   - Learned how to use APIs like Yahoo Finance to collect historical stock data.
   - Understood the importance of preprocessing data in machine learning, such as scaling the data to a suitable range using MinMaxScaler, which can improve the performance of certain algorithms.
   - Learned about creating suitable datasets for time-series prediction, with a particular focus on how to structure data for LSTM models.

2. **Time Series Analysis and LSTM Models**:
   - Gained a deeper understanding of time-series analysis and prediction, especially in the context of stock price data.
   - Learned how to implement and train LSTM models using TensorFlow, a powerful library for deep learning. This included understanding the architecture of LSTM models and the importance of parameters like the number of neurons and layers.
   - Understood the importance of reshaping input data to 3D for LSTM models.

3. **Model Evaluation and Error Metrics**:
   - Learned about different error metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), and understood their relevance in evaluating the performance of prediction models.

4. **Visualizing and Interpreting Results**:
   - Gained experience in visualizing time-series data and prediction results using matplotlib and seaborn, which are critical for understanding and interpreting the model's performance.
   - Understood the importance of comparing predicted results with actual data, both visually and numerically (in a table), to evaluate the model's effectiveness.

5. **Practical Implications and Limitations**:
   - Recognized the practical applications of machine learning in predicting stock prices, which has potential uses in algorithmic trading, portfolio management, risk management, etc.
   - Learned about the inherent challenges and limitations in predicting stock prices due to the complex, stochastic nature of financial markets. This highlights the importance of using machine learning predictions as one of many tools in decision-making, rather than the sole basis for investment decisions. 

## Conclusion
This project demonstrates the use of LSTM models for time-series prediction, specifically for predicting stock prices. It shows that LSTM models can capture the temporal dependencies in stock price data and make reasonably accurate predictions. However, stock price prediction is inherently difficult due to the many factors that can influence prices. Therefore, these predictions should be used with caution and not be the sole basis for investment decisions.