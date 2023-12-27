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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import xgboost as xgb


def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i : (i+sequence_length)]
        sequences.append(sequence)
    return np.array(sequences)

def wdatee():
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

    #create formatted stocks but with data - this is 
    #for visiulization and not actual stuffski
    stocks_w_date = stock_data.copy()
    exclude_columns = ['id', 'day_of_week', 'month', 'year']
    stocks_w_date = stocks_w_date.drop(columns=exclude_columns)

    #make stocks with date numeric
    numeric_columns = [col for col in stocks_w_date.columns if col in ['id', 'date']]
    stocks_w_date[numeric_columns] = stocks_w_date[numeric_columns].apply(pd.to_numeric, errors='coerce')

    moving_averages_df = pd.DataFrame()
    moving_averages_dict = {}

    for column in formatted_stocks:
        moving_averages_dict[f'ma_{column}'] = formatted_stocks[column].rolling(window=5).mean()

    # Concatenate the original DataFrame with the new DataFrame containing moving averages
    #moving_averages_df = pd.concat([moving_averages_df, moving_averages], axis=1)
    moving_averages_df = pd.DataFrame(moving_averages_dict)

    #drop nas for all
    moving_averages_df = moving_averages_df.dropna()
    formatted_stocks = formatted_stocks.dropna()
    stocks_w_date = stocks_w_date.dropna()



    #NOW DO AI STUFF HEHE
    target_col = "ma_open_0900"
    target_col = "open_0900"

    #now I'm trying to predict the end of a day
    #target_col = "close_2205"


    #with raw dataset
    features = formatted_stocks.drop(target_col, axis=1)
    target = formatted_stocks[target_col]

    #HEREHERHERHERHERH WITH DATE
    #take off date bc I don't want it in the calculation.
    #date_column = stocks_w_date.pop('date')
    features = stocks_w_date.drop(target_col, axis=1)
    target = stocks_w_date[target_col]
    #optimus prime
    #print(stocks_w_date)

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)
    #random state is set to a number, I'm using 42 but let's experiment lol

    #better split, more chronological
    split_date = int(len(features) * 0.8)
    X_train, X_test = features[:split_date], features[split_date:]
    y_train, y_test = target[:split_date], target[split_date:]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    #take off the date column for the calculation, but store it
    X_train_date_col = X_train.pop('date')
    X_test_date_col = X_test.pop('date')
    stocks_w_date_date_col = stocks_w_date.pop('date')

    # Create and train the model
    #model = LinearRegression()
    model = xgb.XGBRFRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')


    #add back in date to X_test for visualization
    #ONLY FOR VISUALIZATION NOT ANY CALCULATION !!!!
    stocks_w_date = pd.concat([stocks_w_date, stocks_w_date_date_col], axis=1)
    X_test = pd.concat([X_test, X_test_date_col], axis=1)
    X_test['date'] = pd.to_datetime(X_test['date'])
    X_test['date'] = X_test['date'].dt.strftime('%m-%d')
    stocks_w_date['date'] = pd.to_datetime(stocks_w_date['date'])
    stocks_w_date['date'] = stocks_w_date['date'].dt.strftime('%m-%d')

    

    #JUST SHOWING IMPORTANCE
    #Get feature importance
    importance = model.get_booster().get_fscore()
    # Create a DataFrame from the dictionary of feature importance
    feature_importance = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.scatter(stocks_w_date['date'], stocks_w_date[target_col], color='blue', label='Actual Prices')
    plt.scatter(X_test['date'], predictions, color='red', label='Predicted Prices')
    plt.title('Stock Prices over Time')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


