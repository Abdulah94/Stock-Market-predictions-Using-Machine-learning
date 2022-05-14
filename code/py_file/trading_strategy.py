#  this file mainly used to design trading strategy

from data_access import data
from prophet.plot import add_changepoints_to_plot
import yfinance as yf
import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt


data['ds'] = data.index
data_prophet = data[['ds', 'Close']]
data_prophet = data_prophet.rename(columns = {'Close': 'y'})

m = Prophet()
dataframe = data_prophet 
periods_q = int(input("How long you want to use Prophet to estimate stock price (in days)? "))

m.fit(dataframe)
future = m.make_future_dataframe(periods = periods_q)
forecast = m.predict(future)
stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
data_combine = pd.merge(dataframe, stock_price_forecast, on = 'ds')
data_combine['real price percent change'] = data_combine['y'].pct_change()
data_combine['real price percent change'] = data_combine['real price percent change'].replace(np.nan, 0)


def estimation_result(dataframe, periods):
    '''this function shows the estimated stock price by line chart rather than shaow area'''
    '''the blue line is the real price, the yellow line is the estimated price,the green line and red line would be the estimated lower level and upper level respectively.'''
   
    data_combine.set_index('ds')[['y', 'yhat', 'yhat_lower', 'yhat_upper']].plot(color=['royalblue', "yellow", "green", "red"], grid=True)


# Trading Strategy 


# Hold: Our bench mark. Simplest trading strategy, when we predict a stock will rise in price, we hold until the last day of the prediction. Any trading strategy is compared based on this trading strategy. In this case, our theoretical calculation is based primarily on the magnitude of the actual price volatility, which is a cumulative volatility range, and I use '.cumprod()' to calculate it and make a separate column to see the investor's return on unit assets, over a given time period.

def f_hold(row):
    if abs(row['real price percent change']) > 1 :
        val = 'sell'
    elif row['real price percent change'] == 0:
        val = 'buy'
    else:
        val = 'hold'
    return val


def holding_strategy_table():
    '''This function create a table about holding strategy, include when we purchase, when we sold.'''
    holding_sum = data_combine
    holding_sum['Hold_return on unit assets'] = (holding_sum['real price percent change'] + 1).cumprod()
    holding_sum['suggested strategy'] = holding_sum.apply(f_hold, axis=1)
    holding_sum_end = holding_sum[['ds', 'real price percent change','Hold_return on unit assets', 'suggested strategy']]
    return holding_sum_end






# Prophet: This strategy is to sell when our forecast indicates a down trend and buy back in when it indicates an upward trend. In this case, I set that when prophet forecasts the closing price for this stock (y_hat) in the next day( t+1 day) is less than the closing price (y) of today (t day), I will sell all stocks, and then use the money I get to buy all the stocks on the next trading day (t+1 day). 
def f_prophet(row):
    if row['yhat'].shift(-1) > row['yhat']:
        val = 'sell and rebuy it tomorrow'
    elif row['real price percent change'] == 0:
        val = 'buy'
    else:
        val = 'hold'
    return val


def prophet_strategy_table():
    
    prophet_sum = data_combine
    prophet_sum['Prophet Strategy return on unit assets'] = ((prophet_sum['yhat'].shift(-1) > prophet_sum['yhat']).shift(1) * (prophet_sum['real price percent change']) + 1).cumprod()
    
    prophet_sum.loc[prophet_sum['yhat'].shift(-1) > prophet_sum['yhat'], 'suggested strategy'] = "sell and rebuy it tomorrow"
    prophet_sum.loc[prophet_sum['real price percent change'] == 0, 'suggested strategy'] = "by"
    prophet_sum.loc[prophet_sum['yhat'].shift(-1) <= prophet_sum['yhat'], 'suggested strategy'] = "hold"
    # prophet_sum['suggested strategy'] = prophet_sum.apply(f_prophet, axis=1,)
    prophet_sum_end = prophet_sum[['ds', 'real price percent change','Prophet Strategy return on unit assets', 'suggested strategy']]
    return prophet_sum_end
print(prophet_strategy_table())



# Prophet Thresh: This strategy is to only sell when the stock price fall below our yhat_lower boundary.

def prophet_thresh_strategy_table():
    prophet_thresh_sum = data_combine
    prophet_thresh_sum['Prophet Thresh Strategy return on unit assets']  = ((prophet_thresh_sum['y'] > prophet_thresh_sum['yhat_lower']).shift(1)* (prophet_thresh_sum['real price percent change']) + 1).cumprod()
    
    prophet_thresh_sum.loc[prophet_thresh_sum['y'] > prophet_thresh_sum['yhat_lower'], 'suggested strategy'] = "sell and rebuy it tomorrow"
    prophet_thresh_sum.loc[prophet_thresh_sum['y'] <= prophet_thresh_sum['yhat_lower'], 'suggested strategy'] = "hold"
    prophet_thresh_sum.loc[prophet_thresh_sum['real price percent change'] == 0, "suggested strategy"] = 'buy'
    prophet_thresh_sum_end = prophet_thresh_sum[['ds', 'real price percent change','Prophet Thresh Strategy return on unit assets', 'suggested strategy']]
    return prophet_thresh_sum_end






# Seasonality: This strategy is to exit the market in August and re-enter in Ocober, which means the invester will sold all stocks in the end of August, and repurchase the stocks with all initial money at the beginning of October. This was based on the seasonality chart from above.







