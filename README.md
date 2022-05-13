# eco395_final_project

## Reproducibility

**To do so:**
1. First, download the TA-lib library by running the below command in a single jupyter cell. It will take a couple minutes to complete. This library provides different technical indicators. 
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


## Limitations and room for improvements

The current classification model only uses some micro-level variables to predict the stock movement based on volatility and momentum, adding additional macro-level indicators with extra advanced features engineering will improve the model performance. Some stocks were harder to predict with a reliable accuracy rate. 
