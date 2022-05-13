# eco395_final_project

## Description and methodology

In this project we will conduct Technical Analysis and Machine Learning models on S&P500 daily stock data, to help traders to take better decisions by gaining insights on past price patterns and predicting future movements.

We will start with technical analysis by using mean reversion trading strategies (RSI, Bollinger Bands) and momentum-based strategy (ADX), then we’ll move to the ML models to predict future movements.
In this part we’ll start with a classification model by different predictive variables and technical indicators. Then …..

## SQL dataset
In this project we used a sql dataset that obtains and store stock data from Yahoo Finance. To create your own SQL dataset refer to the following instructions ["SQL Setup"](https://github.com/Abdulah94/eco395_final_project/blob/main/code/sql_setup/README.md)


## Reproducibility

**To do so:**
1. First, set up your SQL database by following the link above.
2. Install required libraries before running the code by typing the following command in your terminal, ```Python
pip install -r requirements.txt```
3. Download the TA-lib library by running the below command in a single jupyter cell. It will take a couple minutes to complete. This library provides different technical indicators. 
--- 
```python
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

!tar -xzf ta-lib-0.4.0-src.tar.gz
%cd ta-lib/
!./configure --prefix=$HOME
!make
!make install

!TA_LIBRARY_PATH=~/lib TA_INCLUDE_PATH=~/include pip install ta-lib
```
--- 
4. Make sure to change the queries and functions variable names in the code to the stock ticker you to perform the analysis to.

## Technical Analysis
Technical analysis is a trading discipline employed to evaluate investments and identify trading opportunities by analyzing statistical trends gathered from trading activity, such as price movement and volume. 

##### Mean Reversion Trading Strategies
Mean reversion trading uses signals to detect market imbalance and takes long positions in an oversold market and short positions in an overbought market.
Relative Strength Index (RSI):
RSI measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.
Typically, an RSI over 70 indicates an overbought market condition, which means the asset is overvalued and the price may reverse. An RSI below 30 suggests an oversold market condition, which means the asset is undervalued and the price may rally.
Blinger Bands (BB):
Bollinger Bands are a popular type of volatility indicator, they help determine whether prices are high or low on a relative basis. For example, If the price moves very close to the upper band, it suggests the price is relatively too expensive, and vice-versa.

##### Momentum-based strategies
Momentum-based strategies believe that "the trend is your friend," and use signals to indicate the trend and profit by riding it.
Average Directional Movement Index (ADX):
ADX is used by traders to quantify the overall strength of a trend, but it does not tell you the direction of a trend.



## ML Classification Model for Stock Market Prediction

In this part of the project, we want to build a short-term investment strategy that could be used by daily traders, along with the other techniques and strategies that we have covered before. The goal is to use machine learning tools to assist traders in making their daily trading decisions; specifically, we intend to utilize a classification model to predict the direction of the stock movement for each trading day. The approach follows the momentum trading methodology to predict the stock based on the volatility, volume, and price strength. Accordingly, the features engineering we did is around this thought. We selected different predictive variables and technical indicators, some of which have been discussed earlier; we list some of them below:

1. Relative strength index (RSI)
2. Simple moving average (SMA)
3. Average directional index (ADX)
4. Stop and reverse indicator (SAR)
5. Correlation between the closing price and the moving average
6. The previous high, low, and close prices

Next, and after adding the selected features, we add another column for the dependent variable that we would like to predict, which is the "Signal" variable, and it has three different values, "Buy", "Sell", and "Hold". The signals will be defined based on the distribution of the stock's historical daily returns, and the assumption here is that the stock's daily returns follow a normal distribution. The returns will have negative values (losses) and positive values (gains). If the predicted value of return falls below the 40th quantile of the historical returns’ distribution, the signal will be to sell because the model predicts a loss; similarly, if the predicted return value lies above the 60th quantile, the signal will be to buy as the model predicts a possible gain. A simple draw is given below to clarify the idea.

** A picture goes here. **

Finally, after we trained the classification model, we tested its performance on different S&P500 stocks. The model's performance varied from one stock to another. Some stocks got a high accuracy rate ( > 50% ), and some other stocks were harder to accurately predict ( < 50% ) and had a low accuracy rate. However, out of 15 randomly selected stocks, we got an average accuracy rate of 58%. Below is the confusion matrix and the classification report for the model's performance in predicting Google's stock movement. 

** A picture goes here **

### Machine learning regression model using PROPHET package
In this part we used the Facebook prophet package in addition to various functions to design three basic stock trading models according to the calculation results, these methods proved the validity of the trading model according to the historical data backtesting.


## Limitations and room for improvements

The current classification model only uses some micro-level variables to predict the stock movement based on volatility and momentum, adding additional macro-level indicators with extra advanced features engineering will improve the model performance. Some stocks were harder to predict with a reliable accuracy rate. 
