import requests
import json
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
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



    #STATS MODELING STUFF
    # Assuming 'ma_close_2205' is the target column
    target_column = 'ma_close_2205'
    y = moving_averages_df[target_column]

    # Exclude the target column from the list of all columns in moving_averages_df
    additional_features = [col for col in moving_averages_df.columns if col != target_column]

    # Concatenate the target column and additional features
    #features = pd.concat([y] + [formatted_stocks[col] for col in additional_features], axis=1)
    features = moving_averages_df

    # Train-test split
    train_size = int(len(y) * 0.8)
    train, test = features[:train_size], features[train_size:]

    # Fit a SARIMA model
    order = (1, 1, 1)  # Replace with appropriate order
    seasonal_order = (1, 1, 1, 12)  # Replace with appropriate seasonal order

    model = SARIMAX(train[target_column], exog=train[additional_features], order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)

    # Make predictions on the test set
    predictions = result.get_forecast(steps=len(test), exog=test[additional_features])
    predicted_mean = predictions.predicted_mean

    # Evaluate the model
    mse = mean_squared_error(test[target_column], predicted_mean)
    print(f'Mean Squared Error: {mse}')

    # Plot the actual vs predicted values
    import matplotlib.pyplot as plt
    plt.plot(y.index, y, label='Actual')
    plt.plot(test.index, predicted_mean, label='Predicted', color='red')
    plt.legend()
    plt.show()






    exit()

    #NOW ONTO AI STUFF JEEZ LOUISE
    #y = formatted_stocks['close_0900']
    #y = moving_averages_df['ma_close_2205']
    target_col = "ma_close_2205"
    y = moving_averages_df[target_col]
    additional_features = [col for col in moving_averages_df.columns if col != target_col]
    #print(y)
    #exit()

    # Train-test split
    train_size = int(len(y) * 0.8)
    #train_size = int(len(y) * 0.9)
    train, test = y[:train_size], y[train_size:]
    #print(train)
    #print("ASD:FKADSJ")
    #print(test)
    #exit()

    # Fit a SARIMA model
    order = (1, 1, 1)  # Replace with appropriate order
    seasonal_order = (1, 1, 1, 12)  # Replace with appropriate seasonal order

    model = SARIMAX(train[target_col], exog=train[additional_features], order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)

    # Make predictions on the test set
    #predictions = result.get_forecast(steps=len(test))
    predictions = result.get_forecast(steps=len(test), exog=test[additional_features])
    predicted_mean = predictions.predicted_mean

    # Evaluate the model
    mse = mean_squared_error(test, predicted_mean)
    print(f'Mean Squared Error: {mse}')


    # Plot the actual vs predicted values
    import matplotlib.pyplot as plt
    plt.plot(y.index, y, label='Actual')
    plt.plot(test.index, predicted_mean, label='Predicted', color='red')
    plt.legend()
    plt.show()

    exit()












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