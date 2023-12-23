import requests
import json
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow import LTSM
from tensorflow import keras
from keras.layers import Dense
from keras import layers
from layers import LTSM
from layers import Dense
from keras import models
import tensorflow as tf
from tf.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def getdfs():
    print("STONKZ")
    conn = sqlite3.connect('appl1.db')
    query = "SELECT * FROM appl1"
    stock_data = pd.read_sql_query(query, conn)
    
    #just in case weird formatting
    stock_data = stock_data.dropna()

    #drop unix time
    columns_to_drop = [col for col in stock_data.columns if 'unix_time' in col]
    stock_data = stock_data.drop(columns=columns_to_drop)

    #format date
    stock_data['date'] = pd.to_datetime(stock_data['date'], format='%m/%d/%Y')
    stock_data = stock_data.sort_values(by='date')
    stored_stock_data = stock_data

    # List of columns to apply moving average (excluding 'date' and 'id')
    columns_to_smooth = [col for col in stock_data.columns if col not in ['date', 'id']]

    stock_data['day_of_week'] = stock_data['date'].dt.dayofweek
    stock_data['month'] = stock_data['date'].dt.month
    stock_data['year'] = stock_data['date'].dt.year

    #convert all to numeric
    numeric_columns = [col for col in stock_data.columns if col not in ['id', 'date']]
    stock_data[numeric_columns] = stock_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    #calculate daily return and add that back as a column
    daily_return = stock_data['close_0900'].pct_change()
    stock_data = pd.concat([stock_data, daily_return.rename('daily_return')], axis=1)

    #create formatted_stocks, which is more numbers to calculate moving averages
    exclude_columns = ['id', 'date', 'day_of_week', 'month', 'year']
    formatted_stocks = stock_data.copy()
    formatted_stocks = formatted_stocks.drop(columns=exclude_columns)

    moving_averages_df = pd.DataFrame()
    moving_averages_dict = {}

    for column in formatted_stocks:
        moving_averages_dict[f'ma_{column}'] = formatted_stocks[column].rolling(window=5).mean()

    # Concatenate the original DataFrame with the new DataFrame containing moving averages
    #moving_averages_df = pd.concat([moving_averages_df, moving_averages], axis=1)
    moving_averages_df = pd.DataFrame(moving_averages_dict)

    moving_averages_df = moving_averages_df.dropna()

    #drop na for formatted_stocks
    formatted_stocks = formatted_stocks.dropna()
    print(moving_averages_df)



    #NOW DO AI STUFF HEHE