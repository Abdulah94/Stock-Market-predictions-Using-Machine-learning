# in this file, I want to use facebook prophet do some estimation, and it mainly report some plots about the stock user choose.

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

def future_price_prediction(dataframe, periods):
    '''let users choose how long they wnat to use Prophet to estimate, it will give use the latest result as the predicted table is too long.'''
    m.fit(dataframe)
    future = m.make_future_dataframe(periods = periods_q)
    forecast = m.predict(future)
    latest_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    return latest_pred


def Prophet_prediction_plot(dataframe, periods):
    '''show the prediction result compare to real price'''
    m = Prophet()
    m.fit(dataframe)
    future = m.make_future_dataframe(periods = periods_q) 
    forecast = m.predict(future)
    m.plot(forecast)
    m.plot_components(forecast)


# Trend Changepoints
# real time series frequently have abrupt changes in their trajectories. By default, Prophet will automatically detect these changepoints and will allow the trend to adapt appropriately

#Automatic changepoint detection in Prophet
def auto_changepoint_detection(dataframe, period):
    '''Automatic changepoint detection in Prophet'''
    m.fit(dataframe)
    future = m.make_future_dataframe(periods = periods_q)
    forecast = m.predict(future)
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)

# do I need to include cross validation in this part?
