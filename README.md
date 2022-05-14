# Stock Market predictions Using Machine learning

## Introduction

In this project, we will conduct Technical Analysis and Machine Learning models on S&P500 daily stock data to help traders make better decisions by gaining insights into past price patterns and predicting future movements.

We will start with technical analysis using mean reversion trading strategies (RSI, Bollinger Bands) and momentum-based strategy (ADX). Then we'll move to the ML models to predict future movements.
In this part, we'll start with a classification model to classify the movements of stock prices and give a recommendation based on the distribution of the historical returns. Finally, we'll conduct a regression-based machine learning model using the "Prophet" package.

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

## Methodology

### Technical Analysis
Technical analysis is a trading discipline employed to evaluate investments and identify trading opportunities by analyzing statistical trends from trading activity, such as price movement and volume. 

##### Mean Reversion Trading Strategies
Mean reversion trading uses signals to detect market imbalance and takes long positions in an oversold market and short positions in an overbought market.
Relative Strength Index (RSI):
RSI measures the magnitude of recent price changes to evaluate overbought or oversold conditions in a stock or other asset price.
Typically, an RSI over 70 indicates an overbought market condition, which means the asset is overvalued, and the price may reverse. An RSI below 30 suggests an oversold market condition, which means the asset is undervalued, and the price may rally.


Bollinger Bands (BB):
Bollinger Bands are a popular type of volatility indicator; they help determine whether prices are high or low on a relative basis. For example, If the price moves very close to the upper band, it suggests the price is relatively too expensive, and vice-versa.

##### Momentum-based strategies
Momentum-based strategies believe that "the trend is your friend" and use signals to indicate the trend and profit by riding it.
Average Directional Movement Index (ADX):
Traders use ADX to quantify the overall strength of a trend, but it does not tell you the direction of a trend.



## ML Classification Model for Stock Market Prediction

In this part of the project, we want to build a short-term investment strategy that could be used by daily traders, along with the other techniques and strategies that we have covered before. The goal is to use machine learning tools to assist traders in making their daily trading decisions; specifically, we intend to utilize a classification model to predict the direction of the stock movement for each trading day. The approach follows the momentum trading methodology to predict the stock based on the volatility, volume, and price strength. Accordingly, the features engineering we did is around this thought. We selected different predictive variables and technical indicators, some of which have been discussed earlier; we list some of them below:

1. Relative strength index (RSI)
2. Simple moving average (SMA)
3. Average directional index (ADX)
4. Stop and reverse indicator (SAR)
5. Correlation between the closing price and the moving average
6. The previous high, low, and close prices

Next, and after adding the selected features, we add another column for the dependent variable that we would like to predict, which is the "Signal" variable, and it has three different values, "Buy", "Sell", and "Hold". The signals will be defined based on the distribution of the stock's historical daily returns, and the assumption here is that the stock's daily returns follow a normal distribution. The returns will have negative values (losses) and positive values (gains). If the predicted value of return falls below the 40th quantile of the historical returns’ distribution, the signal will be to sell because the model predicts a loss; similarly, if the predicted return value lies above the 60th quantile, the signal will be to buy as the model predicts a possible gain. 


Finally, after we trained the classification model, we tested its performance on different S&P500 stocks. The model's performance varied from one stock to another. Some stocks got a high accuracy rate ( > 50% ), and some other stocks were harder to accurately predict ( < 50% ) and had a low accuracy rate. However, out of 15 randomly selected stocks, we got an average accuracy rate of 58%. Below is the confusion matrix and the classification report for the model's performance in predicting Amazon's stock movement. 

```python
classification_model_evaluation(amzn)
```

    None               precision    recall  f1-score   support
    
             Buy       0.77      0.86      0.81        56
            Hold       0.50      0.32      0.39        28
            Sell       0.78      0.83      0.81        65
    
        accuracy                           0.74       149
       macro avg       0.69      0.67      0.67       149
    weighted avg       0.73      0.74      0.73       149
    


###### The model's accuracy in predicting Amazon's stock movement direction is 74%. 

### Machine learning regression model using PROPHET package
In this part, we used the Facebook prophet package and various functions to design three basic stock trading models according to the calculation results. These methods proved the validity of the trading model according to the historical data backtesting.

In this model, we used an API package called "yfinance" to obtain the latest data of each stock, and create a user-interactive python file, to be able to freely select the stocks we want to carry out the analysis.

We've created various graphs to analyze the development prospects of each stock before we decide to invest in it. Then, we designed three stock trading methods to compare them with the primary holding method and finally came to a reliable conclusion, according to the results predicted by the prophet, we can make more profits than ordinary strategies.

###### We will use this model to predict and backtest the price of Google from 2020-01-01 up to 2022-04-29.

##### Strategy 1 
Hold strategy (Our benchmark); we use this strategy as a benchmark to evaluate the performance of other trading strategies. When we predict the stock price will increase, we buy and hold until the last day of the prediction period.



##### Strategy 2 
"Prophet" this strategy is to sell when our forecast indicates a downtrend and buy back our position when it shows an upward trend. In this case, when the prophet forecasts the closing price for this stock (y_hat) in the next day( t+1 day) is less than the estimated price (y_hat) of today (t day), we will sell all stocks, and then repurchase our current position the next trading day (t+1 day).


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>real price percent change</th>
      <th>Prophet Strategy return on unit assets</th>
      <th>Prophet Strategy index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-02</td>
      <td>29.090000</td>
      <td>30.477072</td>
      <td>27.897604</td>
      <td>33.076772</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-03</td>
      <td>27.650000</td>
      <td>30.234126</td>
      <td>27.498077</td>
      <td>32.759491</td>
      <td>-0.049502</td>
      <td>1.0</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-06</td>
      <td>27.320000</td>
      <td>29.741680</td>
      <td>26.985261</td>
      <td>32.415682</td>
      <td>-0.011935</td>
      <td>1.0</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-07</td>
      <td>27.219999</td>
      <td>29.619351</td>
      <td>27.177254</td>
      <td>32.105663</td>
      <td>-0.003660</td>
      <td>1.0</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-08</td>
      <td>27.840000</td>
      <td>29.481897</td>
      <td>27.066887</td>
      <td>32.037029</td>
      <td>0.022777</td>
      <td>1.0</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>530</th>
      <td>2022-01-26</td>
      <td>NaN</td>
      <td>17.853920</td>
      <td>15.497142</td>
      <td>20.493468</td>
      <td>0.000000</td>
      <td>1.452362</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>531</th>
      <td>2022-01-27</td>
      <td>NaN</td>
      <td>17.763839</td>
      <td>15.252311</td>
      <td>20.516857</td>
      <td>0.000000</td>
      <td>1.452362</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>532</th>
      <td>2022-01-28</td>
      <td>NaN</td>
      <td>17.680346</td>
      <td>15.030545</td>
      <td>20.356985</td>
      <td>0.000000</td>
      <td>1.452362</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>533</th>
      <td>2022-01-29</td>
      <td>NaN</td>
      <td>16.781794</td>
      <td>14.307839</td>
      <td>19.252044</td>
      <td>0.000000</td>
      <td>1.452362</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>534</th>
      <td>2022-01-30</td>
      <td>NaN</td>
      <td>16.763431</td>
      <td>14.086911</td>
      <td>19.241965</td>
      <td>0.000000</td>
      <td>1.452362</td>
      <td>buy</td>
    </tr>
  </tbody>
</table>
<p>535 rows × 8 columns</p>
</div>





##### Strategy 3
"Prophet Thresh", this strategy follows a more cautious approach. If the actual closing price of the previous day (day t-1) is lower than the low value we predicted the next day (day t), we will sell all of our position on day (day t-1) and repurchase it the next day(day t).


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>real price percent change</th>
      <th>Prophet Thresh Strategy return on unit assets</th>
      <th>suggested strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-02</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>buy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-03</td>
      <td>-0.004907</td>
      <td>1.000000</td>
      <td>hold</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-06</td>
      <td>0.024657</td>
      <td>1.000000</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-07</td>
      <td>-0.000624</td>
      <td>0.999376</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-08</td>
      <td>0.007880</td>
      <td>1.007251</td>
      <td>sell and rebuy it tomorrow</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>611</th>
      <td>2022-05-24</td>
      <td>0.000000</td>
      <td>0.902095</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>612</th>
      <td>2022-05-25</td>
      <td>0.000000</td>
      <td>0.902095</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>613</th>
      <td>2022-05-26</td>
      <td>0.000000</td>
      <td>0.902095</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>614</th>
      <td>2022-05-27</td>
      <td>0.000000</td>
      <td>0.902095</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>615</th>
      <td>2022-05-28</td>
      <td>0.000000</td>
      <td>0.902095</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>616 rows × 4 columns</p>
</div>




##### Strategy evaluation

```

    Hold = 1,747
    Prophet = 6,098
    Prophet Thresh = 902



```

We can see that by following the "Prohet" strategy we'll generate much higher profits compared to just buy and hold the stock.




## Limitations and room for improvements

In all of these models, we only used momentum variables (price, volume, ...etc); incorporating fundamental indicators and macro-level analysis in our models will increase the reliability of our predictions.


