# Forecasting Bitcoin Prices with Time Series Analysis

## Introduction
Time series forecasting is a powerful tool used to predict future values based on previously observed data points. In financial markets, such as Bitcoin (BTC), accurate forecasting helps investors and traders make informed decisions. Given the volatile nature of cryptocurrencies, time series forecasting can be particularly useful in predicting future BTC price movements, thereby providing insights into potential market trends.


## An Introduction to Time Series Forecasting
Time series forecasting involves using past data to predict future values. It is widely used in financial markets to predict stock prices, interest rates, and, in this case, cryptocurrency prices like Bitcoin (BTC). Forecasting BTC prices is valuable because it helps traders manage risks and find trading opportunities in a highly volatile market.


## Preprocessing Method
Before training the model, the data was cleaned and normalized. Missing values were removed, and the 'Close' prices of BTC were scaled to a range of 0-1 using MinMaxScaler. This ensures that the model can learn effectively from the data.


## Setting Up tf.data.Dataset for Model Inputs
The dataset was structured using past BTC prices to predict future prices. A window size of 60 days was chosen, meaning the model looks at the past 60 days to make a prediction for the next day. TensorFlow's `tf.data.Dataset` API was used to prepare the data for training.


## Model Architecture
An LSTM-based neural network was used for forecasting. LSTM is effective at capturing time dependencies in sequential data. The model consists of two LSTM layers followed by Dense layers to predict BTC prices. The Adam optimizer was used, and the model was trained using mean squared error as the loss function.


## Results and Evaluation
The model's performance was evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The graph below shows predicted vs. actual BTC prices, where we observe that the model successfully captures general trends, although there may be some discrepancies during volatile periods.


## Conclusion
This project provided insight into the challenges and possibilities of forecasting Bitcoin prices. While the model effectively captured general trends, it faced challenges during periods of high volatility. This suggests that more sophisticated models or additional features could further improve forecasting accuracy.
