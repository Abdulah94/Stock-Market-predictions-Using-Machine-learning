# ##  below is my final project's material

# ## I will use the facebook.prophet package to make predictions for selected stock price, and based on its prediction to design the trading strategy

import yfinance as yf
import pandas as pd
from prophet import Prophet
import numpy as np

import re

regex = re.compile("[0-9]{4}\-[0-9]{2}\-[0-9]{2}")

def check_date_format(date):
    match = re.match(regex, date)
    
    if (match):
        return date
    elif date == 'q':
        return None
    else: 
       return check_date_format(input("Invalid date, try again or press 'q' to quit: "))


def get_initial_data(stockname, start_date, end_date):
    '''for start date and end date, means the start and end dates of the stocks you want to collect'''
    # stockname = stockname    
    # start_date = select_start_date
    # end_date = select_end_date
    data = yf.download(stockname, start = start_date, end = end_date)
    return data

stockname = input('Please select your stock and make sure the name is correct:')
select_start_date = check_date_format(input('Please select the start date of the stocks you would like to collect: in YYYY-MM-DD format'))
select_end_date = check_date_format(input('Please select the end date of the stocks you would like to collect: in YYYY-MM-DD format'))
stockname = stockname    
start_date = select_start_date
end_date = select_end_date

data = get_initial_data(stockname, start_date, end_date)
