## Stock Market predictions Using Machine learning

#### Importing Libraries


```python
%load_ext lab_black
```


```python
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import talib as ta
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import import_ipynb
from database import engine
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import re
import bt
```

### Queries


```python
query_1 = """
select
	"Date" as "Date",
	"AAPL" as "Close"
from
	close_price cp
order by 
    "Date" desc
    
    """

query_2 = """ 
select 
	v."Date" as "Date",
	op."GOOG" as "Open",
	hp."GOOG" as "High",
	lp."GOOG" as "Low",
	cp."GOOG" as "Close",
	ac."GOOG" as "Adj Close",
	v."GOOG" as "Volume"

FROM 
	close_price cp left join adj_close ac on cp."Date" = ac."Date"
	left join volume v on cp."Date" = v."Date"
	left join open_price op on cp."Date" = op."Date"
	left join high_price hp on cp."Date" = hp."Date"
	left join low_price lp on cp."Date" = lp."Date"

    """
```

### Technical Analysis


```python
# RSI


def RSI(ticker):
    """Calculate RSI"""
    stock_data = pd.read_sql_query(query_1, engine)
    stock_data["RSI"] = ta.RSI(stock_data["Close"], timeperiod=14).to_frame()
    """Create subplots"""
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_ylabel("Price")
    ax1.plot(stock_data["Close"])
    ax2.set_ylabel("RSI")
    ax2.plot(stock_data["RSI"], color="orangered")
    ax1.set_title("Price and RSI")
    plt.show()

    return


print(RSI("AAPL"))
```

    None



```python
# Bolinger Bands
def BB(ticker):
    stock_data = pd.read_sql_query(query_1, engine)
    # Define the Bollinger Bands with 1-sd
    upper_1sd, mid_1sd, lower_1sd = ta.BBANDS(
        stock_data["Close"], nbdevup=2, nbdevdn=2, timeperiod=20
    )
    # Plot the upper and lower Bollinger Bands
    plt.plot(stock_data["Close"], color="green", label="Price")
    plt.plot(upper_1sd, color="tomato", label="Upper 1sd")
    plt.plot(lower_1sd, color="tomato", label="Lower 1sd")

    # Customize and show the plot
    plt.legend(loc="upper left")
    plt.title("Bollinger Bands (2sd)")
    plt.show()
    return


print(BB("GOOG"))
```

    None



```python
# ADX


def ADX(ticker):
    stock_data = pd.read_sql_query(query_2, engine)
    """Calculate ADX"""
    stock_data["ADX"] = ta.ADX(
        stock_data["High"], stock_data["Low"], stock_data["Close"]
    )
    """Create subplots"""
    fig, (ax1, ax2) = plt.subplots(2)
    """ Plot ADX with the price """
    ax1.set_ylabel("Price")
    ax1.plot(stock_data["Close"])
    ax2.set_ylabel("ADX")
    ax2.plot(stock_data["ADX"], color="red")
    ax1.set_title("Price and ADX")
    plt.show()
    return


print(ADX("GOOG"))
```

    None


### ML Classification Model for Stock Market Prediction


```python
def add_features(df):
    """This function takes the stock dataframe, adds to it new columns of different technical indicators as predictive variables for the model to use."""

    n = 7
    df["RSI"] = ta.RSI(np.array(df["Close"].shift(1)), timeperiod=n)
    df["MA"] = df["Close"].shift(1).rolling(window=n).mean()
    df["Corr"] = df["Close"].shift(1).rolling(window=n).corr(df["MA"].shift(1))
    df["SAR"] = ta.SAR(
        np.array(df["High"].shift(1)), np.array(df["Low"].shift(1)), 0.2, 0.2
    )
    df["ADX"] = ta.ADX(
        np.array(df["High"].shift(1)),
        np.array(df["Low"].shift(1)),
        np.array(df["Open"]),
        timeperiod=n,
    )
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Open_change"] = df["Open"] - df["Open"].shift(1)
    df["Open_close_change"] = df["Open"] - df["Prev_Close"]
    df["Return"] = (df["Open"].shift(-1) - df["Open"]) / df["Open"]
    df["return_lag1"] = df["Return"].shift(1)
    df["return_lag2"] = df["Return"].shift(2)
    df["return_lag3"] = df["Return"].shift(3)
    df = df.dropna()

    return df
```


```python
def add_signals(df):
    """This function takes a dataframe and adds a new column for signals, the signals are based on the distribution of the observed returns of the stock during the
    selected period, a 'Sell' signal will be assigned for all observed returns that lie below the 40th quantile (negative returns) to signal a predicted loss,
    similarly a 'Buy' signal will be assigned for all observed returns that lie above the 60th quantile (positive returns) to signal a predicted gain, and
    a 'Hold' signal otherwise, mostly small percentage changes in returns."""

    df["Signal"] = "Hold"
    df.loc[df["Return"] > df["Return"].quantile(q=0.60), "Signal"] = "Buy"
    df.loc[df["Return"] < df["Return"].quantile(q=0.40), "Signal"] = "Sell"

    return df
```


```python
def get_training_split(df):
    """This function takes a dataframe and returns one dataframe and one pandas series, the returned dataframe conatins the features of the training dataset,
    and the returned series contain the related signals of the training dataset."""

    split = int(len(df) * 0.8)

    features = df.drop(["Close", "Signal", "High", "Low", "Volume", "Return"], axis=1)
    signals = df["Signal"]

    training_features_df = features[:split]
    training_signals_series = signals[:split]

    return training_features_df, training_signals_series
```


```python
def get_testing_split(df):
    """This function takes a dataframe and returns one dataframe and one pandas series, the returned dataframe conatins the features of the testing dataset,
    and the returned series contains the related signals of the testing dataset."""

    split = int(len(df) * 0.8)

    features = df.drop(["Close", "Signal", "High", "Low", "Volume", "Return"], axis=1)
    signals = df["Signal"]

    testing_features_df = features[split:]
    testing_signals_series = signals[split:]

    return testing_features_df, testing_signals_series
```


```python
def prediction_model(trainF, trainS, testF):
    """This function takes two dataframes and one pandas series. We will pass the training features dataframe, training actual signals series,
    and the testing features dataframe that we want the model to predict. The function returns a pandas series of the predicted signals."""

    c = [10, 100, 1000, 10000]
    g = [1e-2, 1e-1, 1e0]
    parameters = {"svc__C": c, "svc__gamma": g, "svc__kernel": ["rbf"]}
    steps = [("scaler", StandardScaler()), ("svc", SVC())]
    pipeline = Pipeline(steps)

    rcv = RandomizedSearchCV(pipeline, parameters, cv=TimeSeriesSplit(n_splits=2))
    rcv.fit(trainF, trainS)
    best_C = rcv.best_params_["svc__C"]
    best_kernel = rcv.best_params_["svc__kernel"]
    best_gamma = rcv.best_params_["svc__gamma"]

    cls = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma)
    ss = StandardScaler()

    cls.fit(ss.fit_transform(trainF), trainS)
    predicted_signals = cls.predict(ss.transform(testF))

    return predicted_signals
```


```python
def plot_confusion_matrix(array):
    """This function takes an array (confusion matrix) and returns a plot to visualise the performance."""

    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax = plt.subplot()
    sns.heatmap(array, annot=True, ax=ax)

    ax.xaxis.set_ticklabels(["Sell", "Hold", "Buy"])
    ax.yaxis.set_ticklabels(["Sell", "Hold", "Buy"])

    return plt.show()
```


```python
def classification_model_evaluation(df):
    """This function takes the stock's dataframe, mutates it in the desired format, splits the dataset into a training and testing datasets, gets the model predictions
    and evaluates it, and then returns the confusion matrix plot and the classification accuracy report (out-of-sample accuracy is based on the testing dataset)."""

    df1 = add_features(df)
    df2 = add_signals(df1)

    trainF, trainS = get_training_split(df2)
    testF, testS = get_testing_split(df2)

    predicted_signals = prediction_model(trainF, trainS, testF)

    confusion_matrix_ = confusion_matrix(testS, predicted_signals)
    confusion_matrix_plot = plot_confusion_matrix(confusion_matrix_)
    out_of_sample_accuracy = classification_report(testS, predicted_signals)

    return print(confusion_matrix_plot, out_of_sample_accuracy)
```

#### Model Evaluation

#####     The model's performance varies from stock to stock, some stocks got a high accuracy rate ( > 50% ) and some other stocks were harder to accuratly predict ( < 50% ) and had a low accuracy rate. However, out of 15 randomly selected stocks we got an average accuracy rate of 58%. We listed four examples below. 

##### 1) Amazon


```python
Amazon_query = """ 
select 
	v."Date" as "Date",
	op."AMZN" as "Open",
	hp."AMZN" as "High",
	lp."AMZN" as "Low",
	cp."AMZN" as "Close",
	ac."AMZN" as "Adj Close",
	v."AMZN" as "Volume"

FROM 
	close_price cp left join adj_close ac on cp."Date" = ac."Date"
	left join volume v on cp."Date" = v."Date"
	left join open_price op on cp."Date" = op."Date"
	left join high_price hp on cp."Date" = hp."Date"
	left join low_price lp on cp."Date" = lp."Date"

    """
```


```python
amzn = pd.read_sql_query(
    Amazon_query, engine
)  # could not pass this to my function because I get a type error, so I used yfinance below.
```


```python
amzn = yf.download("AMZN", start="2019-04-29", end="2022-04-29")
```

    [*********************100%***********************]  1 of 1 completed



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

##### 2) Boeing


```python
Boeing = yf.download("BA", start="2019-04-29", end="2022-04-29")
```

    [*********************100%***********************]  1 of 1 completed



```python
classification_model_evaluation(Boeing)
```

    None               precision    recall  f1-score   support
    
             Buy       0.68      0.79      0.73        63
            Hold       0.27      0.12      0.16        26
            Sell       0.75      0.80      0.77        60
    
        accuracy                           0.68       149
       macro avg       0.57      0.57      0.56       149
    weighted avg       0.64      0.68      0.65       149
    


###### The model's accuracy in predicting Boeing's stock movement direction is 68%. 

##### 3) eBay


```python
ebay = yf.download("EBAY", start="2019-04-29", end="2022-04-29")
```

    [*********************100%***********************]  1 of 1 completed



```python
classification_model_evaluation(ebay)
```

    None               precision    recall  f1-score   support
    
             Buy       0.64      0.98      0.78        55
            Hold       0.17      0.19      0.18        21
            Sell       0.88      0.51      0.64        73
    
        accuracy                           0.64       149
       macro avg       0.57      0.56      0.53       149
    weighted avg       0.69      0.64      0.63       149
    


###### The model's accuracy in predicting eBay's stock movement direction is 63%. 

##### 4) Allstate


```python
allstate = yf.download("ALL", start="2019-04-29", end="2022-04-29")
```

    [*********************100%***********************]  1 of 1 completed



```python
classification_model_evaluation(allstate)
```

    None               precision    recall  f1-score   support
    
             Buy       0.49      0.84      0.62        55
            Hold       0.33      0.27      0.30        33
            Sell       0.76      0.36      0.49        61
    
        accuracy                           0.52       149
       macro avg       0.53      0.49      0.47       149
    weighted avg       0.57      0.52      0.50       149
    


### Stock Market Prediction Using Prophet Package


```python
# data access function 


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

```


```python
def get_prophet_data(data):
    data['ds'] = data.index
    data_prophet = data[['ds', 'Close']]
    data_prophet = data_prophet.rename(columns = {'Close': 'y'})
    return data_prophet



def future_price_prediction(dataframe, periods):
    '''let users choose how long they wnat to use Prophet to estimate, it will give use the latest result as the predicted table is too long.'''
    m.fit(dataframe)
    future = m.make_future_dataframe(periods = periods_q)
    forecast = m.predict(future)
    latest_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    return latest_pred


```


```python
# prediction plot, shadow area shows the estimated interval
def Prophet_prediction_plot(dataframe, periods):
    '''show the prediction result compare to real price'''
    m = Prophet()
    m.fit(dataframe)
    future = m.make_future_dataframe(periods = periods_q) 
    forecast = m.predict(future)
    m.plot(forecast)
    m.plot_components(forecast)

```


```python
# show the prediction result with line chat

# this code I wish I can run in sql, but I failed
# future = m.make_future_dataframe(periods = periods_q)
# forecast = m.predict(future)
# stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# data_combine = pd.merge(dataframe, stock_price_forecast, on = 'ds', how='outer')
# data_combine_inner = pd.merge(dataframe, stock_price_forecast, on = 'ds')
# data_combine['real price percent change'] = data_combine['y'].pct_change()
# data_combine['real price percent change'] = data_combine['real price percent change'].replace(np.nan, 0)

def prediction_need_table(data):
    m = Prophet()
    m.fit(dataframe)
    future = m.make_future_dataframe(periods = periods_q)
    forecast = m.predict(future)
    stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    data_combine = pd.merge(dataframe, stock_price_forecast, on = 'ds', how='outer')
    data_combine_inner = pd.merge(dataframe, stock_price_forecast, on = 'ds')
    data_combine['real price percent change'] = data_combine['y'].pct_change()
    data_combine['real price percent change'] = data_combine['real price percent change'].replace(np.nan, 0)
    return data_combine 

  


def estimation_result(dataframe):
    '''this function shows the estimated stock price by line chart rather than shadow areas'''
    '''the blue line is the real price, the yellow line is the estimated price,the green line and red line would be the estimated lower level and upper level respectively.'''
   
    dataframe.set_index('ds')[['y', 'yhat', 'yhat_lower', 'yhat_upper']].plot(color=['royalblue', "yellow", "green", "red"], grid=True)




```

### Trade Strategy


```python
# Hold: Our bench mark. Simplest trading strategy, when we predict a stock will rise in price, 
# we hold until the last day of the prediction. 
# Any trading strategy is compared based on this trading strategy. 
# In this case, our theoretical calculation is based primarily on the magnitude of the actual price volatility, 
# which is a cumulative volatility range, 
# and I use '.cumprod()' to calculate it and make a separate column to see the investor's return on unit assets, 
# over a given time period.

def f_hold(row):
    if abs(row['real price percent change']) > 1 :
        val = 'sell'
    elif row['Hold_return on unit assets'] == 1.00000000:
        val = 'buy'
    else:
        val = 'hold'
    return val


def holding_strategy_table(data):
    '''This function create a table about holding strategy, include when we purchase, when we sold.'''
    holding_sum = data.copy()
    holding_sum['Hold_return on unit assets'] = (holding_sum['real price percent change'] + 1).cumprod()
    holding_sum['suggested strategy'] = holding_sum.apply(f_hold, axis=1)
    holding_sum_end = holding_sum[['ds', 'real price percent change','Hold_return on unit assets', 'suggested strategy']]
    return holding_sum_end
```


```python
# Prophet: This strategy is to sell when our forecast indicates a down trend 
# and buy back in when it indicates an upward trend. In this case, 
# I set that when prophet forecasts the closing price for this stock (y_hat) 
# in the next day( t+1 day) is less than the closing price (y_hat) of today (t day), 
# I will sell all stocks, and then use the money I get to buy all the stocks on the next trading day (t+1 day). 


def f_prophet(row):
    if row['Prophet Strategy index'] < 0:
        val = 'sell and rebuy it tomorrow'
    elif row['Prophet Strategy index'] > 0:
        val = 'hold'
    else:
        val = 'buy'
    return val


def prophet_strategy_table(data):
    
    prophet_sum = data.copy()
    prophet_sum['Prophet Strategy return on unit assets'] = ((prophet_sum['yhat'].shift(-1) > prophet_sum['yhat']).shift(1) * (prophet_sum['real price percent change']) + 1).cumprod()
    prophet_sum['Prophet Strategy index'] = prophet_sum['yhat'].shift(-1) - prophet_sum['yhat']
    
    
    # prophet_sum.loc[prophet_sum['yhat'].shift(-1) > prophet_sum['yhat'], 'suggested strategy'] = "sell and rebuy it tomorrow"
    # prophet_sum.loc[prophet_sum['real price percent change'] == 0, 'suggested strategy'] = "buy"
    # prophet_sum.loc[prophet_sum['yhat'].shift(-1) <= prophet_sum['yhat'], 'suggested strategy'] = "hold"
    prophet_sum['suggested Strategy '] = prophet_sum.apply(f_prophet, axis=1,)
    # prophet_sum_end = prophet_sum[['ds', 'real price percent change','Prophet Strategy return on unit assets', 'suggested strategy']]
    return prophet_sum

```


```python
# Prophet Thresh. This strategy is more confident. 
# If the real price of the day (day t) is lower than the low value we predicted the previous day (day t), 
# we will sell it all on the day (day t) and rebuy it again the next day(day t+1) with all the money.
def f_prophet_thresh(row):
    if row['Prophet Thresh Strategy index'] < 0:
        val = 'sell and rebuy it tomorrow'
    elif row['Prophet Thresh Strategy index'] > 0:
        val = 'hold'
    else:
        val = 'lack yesterday real closing price'
    return val

def prophet_thresh_strategy_table(data):
    prophet_thresh_sum = data.copy()
    prophet_thresh_sum['Prophet Thresh Strategy return on unit assets']  = ((prophet_thresh_sum['y'].shift(-1) > prophet_thresh_sum['yhat_lower']).shift(1)* (prophet_thresh_sum['real price percent change']) + 1).cumprod()
    prophet_thresh_sum['Prophet Thresh Strategy return on unit assets'] =  prophet_thresh_sum['Prophet Thresh Strategy return on unit assets'].replace(np.nan,0)
    prophet_thresh_sum['Prophet Thresh Strategy index'] = prophet_thresh_sum['y'].shift(-1) - prophet_thresh_sum['yhat_lower']
    prophet_thresh_sum['suggested Strategy '] = prophet_thresh_sum.apply(f_prophet_thresh, axis=1,)

    # prophet_thresh_sum.loc[prophet_thresh_sum['y'].shift(-1) < prophet_thresh_sum['yhat_lower'], 'suggested strategy'] = "sell and rebuy it tomorrow"
    # prophet_thresh_sum.loc[prophet_thresh_sum['y'].shift(-1) >= prophet_thresh_sum['yhat_lower'], 'suggested strategy'] = "hold"
    # prophet_thresh_sum.loc[prophet_thresh_sum['Prophet Thresh Strategy return on unit assets'] == 0, "suggested strategy"] = 'buy'
    # prophet_thresh_sum_end = prophet_thresh_sum[['ds', 'real price percent change','Prophet Thresh Strategy return on unit assets', 'suggested strategy']]
    return prophet_thresh_sum

```


```python
#  profit comparison between different stategies, suppose our initial invest money is $1000
def strategy_compare(data):
    
    df = data.copy()
    df['Hold'] = (df['real price percent change'] + 1).cumprod()
    df['Prophet'] = ((df['yhat'].shift(-1) > df['y']).shift(1) * (df['real price percent change']) + 1).cumprod()
    df['Prophet Thresh']  = ((df['y'].shift(-1) > df['yhat_lower']).shift(1)* (df['real price percent change']) + 1).cumprod()

    
    return df

```

### Result Show


```python
stockname = input('Please select your stock and make sure the name is correct:')
select_start_date = check_date_format(input('Please select the start date of the stocks you would like to collect: in YYYY-MM-DD format'))
select_end_date = check_date_format(input('Please select the end date of the stocks you would like to collect: in YYYY-MM-DD format'))
stockname = stockname    
start_date = select_start_date
end_date = select_end_date

data = get_initial_data(stockname, start_date, end_date)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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




```python
# Prophet Thresh. This strategy is more confident.
# If the real price of the day (day t) is lower than the low value we predicted the previous day (day t),
# we will sell it all on the day (day t) and rebuy it again the next day(day t+1) with all the money.


def prophet_thresh_strategy_table():
    prophet_thresh_sum = data_combine.copy()
    prophet_thresh_sum["Prophet Thresh Strategy return on unit assets"] = (
        (prophet_thresh_sum["y"] > prophet_thresh_sum["yhat_lower"]).shift(1)
        * (prophet_thresh_sum["real price percent change"])
        + 1
    ).cumprod()
    prophet_thresh_sum[
        "Prophet Thresh Strategy return on unit assets"
    ] = prophet_thresh_sum["Prophet Thresh Strategy return on unit assets"].replace(
        np.nan, 0
    )

    prophet_thresh_sum.loc[
        prophet_thresh_sum["y"] > prophet_thresh_sum["yhat_lower"], "suggested strategy"
    ] = "sell and rebuy it tomorrow"
    prophet_thresh_sum.loc[
        prophet_thresh_sum["y"] <= prophet_thresh_sum["yhat_lower"],
        "suggested strategy",
    ] = "hold"
    prophet_thresh_sum.loc[
        prophet_thresh_sum["Prophet Thresh Strategy return on unit assets"] == 0,
        "suggested strategy",
    ] = "buy"
    prophet_thresh_sum_end = prophet_thresh_sum[
        [
            "ds",
            "real price percent change",
            "Prophet Thresh Strategy return on unit assets",
            "suggested strategy",
        ]
    ]
    return prophet_thresh_sum_end


prophet_thresh_strategy_table()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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




```python
#  profit comparison between different stategies, suppose our initial invest money is $1000
def strategy_compare():

    df = data_combine.copy()
    df["Hold"] = (df["real price percent change"] + 1).cumprod()
    df["Prophet"] = (
        (df["yhat"].shift(-1) > df["y"]).shift(1) * (df["real price percent change"])
        + 1
    ).cumprod()
    df["Prophet Thresh"] = (
        (df["y"] > df["yhat_lower"]).shift(1) * (df["real price percent change"]) + 1
    ).cumprod()

    return df


df = strategy_compare()
print(f"Hold = {df['Hold'].iloc[-1]*1000:,.0f}")
print(f"Prophet = {df['Prophet'].iloc[-1]*1000:,.0f}")
print(f"Prophet Thresh = {df['Prophet Thresh'].iloc[-1]*1000:,.0f}")
```

    Hold = 1,747
    Prophet = 6,098
    Prophet Thresh = 902



```python
# plot the profit comparison, finally we can prove that
df.set_index("ds")[["Hold", "Prophet", "Prophet Thresh"]].plot(
    figsize=(16, 8),
    color=[
        "royalblue",
        "#34495e",
        "yellow",
    ],
    grid=True,
)
```




    <AxesSubplot:xlabel='ds'>




```python

```


```python

```
