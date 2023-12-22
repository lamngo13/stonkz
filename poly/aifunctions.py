import requests
import json
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def stonkz1():
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
    stock_data['close_0900'] = pd.to_numeric(stock_data['close_0900'], errors='coerce')
    stock_data['daily_return'] = stock_data['close_0900'].pct_change()

    print(stock_data)
    exit()


    # Apply a 5-day moving average to each relevant column
    moving_averages = pd.DataFrame()

    for column in columns_to_smooth:
        moving_averages[f'ma_{column}'] = stock_data[column].rolling(window=5).mean()

    # Concatenate the original DataFrame with the new DataFrame containing moving averages
    stock_data = pd.concat([stock_data, moving_averages], axis=1)

    print(stock_data)
    exit()
    #NOW THE AI STUFF JEEZ LOUSE
    # List of columns to train SARIMA models (excluding 'date', 'id', and non-numeric columns)
    columns_to_train = [col for col in stock_data.columns if col not in ['date', 'id']]
    # Dictionary to store trained models
    trained_models = {}

    # Train SARIMA models for each relevant column
    for column_of_interest in columns_to_train:
        model = SARIMAX(train_data[column_of_interest], order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
        model_fit = model.fit()
        trained_models[column_of_interest] = model_fit

    # Evaluate the Models and Make Predictions for Future Steps
    future_steps = 30  # Number of days to predict into the future

    for column_of_interest in columns_to_train:
        predictions = trained_models[column_of_interest].get_forecast(steps=len(test_data)).predicted_mean
        mae = mean_absolute_error(test_data[column_of_interest], predictions)

        # Visualize Results
        plt.plot(test_data['date'], test_data[column_of_interest], label=f'Actual {column_of_interest}')
        plt.plot(test_data['date'], predictions, label=f'Predicted {column_of_interest}', linestyle='--')
        plt.legend()
        plt.show()







    exit()
    # Train-Test Split
    train_size = int(len(stock_data) * 0.8)
    train_data, test_data = stock_data[:train_size], stock_data[train_size:]

    # Example: Training a SARIMA model for one of the columns (e.g., close_0900)
    column_of_interest = 'close_0900'
    model = SARIMAX(train_data[column_of_interest], order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    model_fit = model.fit()

    # Evaluate the Model
    from sklearn.metrics import mean_absolute_error

    predictions = model_fit.get_forecast(steps=len(test_data))
    mae = mean_absolute_error(test_data[column_of_interest], predictions)

    # Make Predictions for Future Steps
    future_steps = 30  # Number of days to predict into the future
    future_predictions = model_fit.get_forecast(steps=future_steps).predicted_mean

    # Visualize Results
    import matplotlib.pyplot as plt

    plt.plot(test_data['date'], test_data[column_of_interest], label='Actual Prices')
    plt.plot(test_data['date'], predictions, label='Predicted Prices', linestyle='--')
    plt.legend()
    plt.show()



    print(stock_data)