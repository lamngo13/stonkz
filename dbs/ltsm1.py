# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta, date
import warnings
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#THIS POPULATES wpred.db with good predictions NONRECURSIVELY

#earliest ik of is 8-23

warnings.filterwarnings("ignore", message=".*A value is trying to be set on a copy of a slice from a DataFrame.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names, but SimpleImputer was fitted with feature names.*", category=UserWarning)


# List of stock tickers
stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']
storer = []

# Connect to the SQLite database
conn = sqlite3.connect('eltee.db')

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

#9 25 to 10 17
startDate = datetime(2023, 9, 25).date()
endDate = datetime(2023, 10, 17).date()

currentDate = startDate

RecieverDate = datetime(2023, 10, 18).date()

iterator = 0

while iterator <= 28:

    #FIRST, check to see if reciever date is valid, and if not, iterate it. 
    goodReciever = False
    existsquery = '''SELECT * FROM ALLONE WHERE date = ?'''
    while (not goodReciever):
        cursor = conn.cursor()
        checker = cursor.execute(existsquery, (RecieverDate,))
        checker = cursor.fetchone()

        if (checker is not None):
            goodReciever = True
        else:
            #we must iterate to the next date 
            RecieverDate += timedelta(days=1)
            #print("not found rip")
    #end while loop
    print("rec date: " + str(RecieverDate))

    #measure dates here:
    print("iterator: " + str(iterator))
    print("start: " + str(currentDate))
    print("end: " + str(endDate))
    print("REC DATE: " + str(RecieverDate))

    # Loop through each stock ticker
    for stock_ticker in stocklist:
        # Query to retrieve stock data for the current stock ticker within a specific date range
        query = f'''
            SELECT date, open, high, low, close, volume
            FROM ALLONE 
            WHERE date BETWEEN '{startDate}' AND '{endDate}' AND name = '{stock_ticker}'
        '''
        # Load data from the database into a pandas DataFrame
        df = pd.read_sql(query, conn)


        #---------------------END RETRIEVAL FROM DB----------------------------
        # Extract date features
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek

        # Create lag features for the previous week's closing price
        df['lag_close_1'] = df['close'].shift(1) 

        # Calculate rolling mean and rolling standard deviation
        window = 3
        df['close_mean'] = df['close'].rolling(window=window).mean()
        df['close_std'] = df['close'].rolling(window=window).std()

        #open
        df['open_mean'] = df['open'].rolling(window=window).mean()
        df['open_std'] = df['open'].rolling(window=window).std()

        #high
        df['high_mean'] = df['high'].rolling(window=window).mean()
        df['high_std'] = df['high'].rolling(window=window).std()
    

        #low
        df['low_mean'] = df['low'].rolling(window=window).mean()
        df['low_std'] = df['low'].rolling(window=window).std()

        #volume
        df['volume_mean'] = df['volume'].rolling(window=window).mean()
        df['volume_std'] = df['volume'].rolling(window=window).std()

        #prediction
        #df['prediction_mean'] = df['prediction'].rolling(window=window).mean()
        #df['prediction_std'] = df['prediction'].rolling(window=window).std()


        # Drop NaN values
        df = df.dropna()

        # Split data into features (X) and target variable (y)
        X = df[['day', 'month', 'day_of_week', 'lag_close_1', 'close_mean', 'close_std', 'open_mean', 'open_std', 'high_mean', 'high_std', 'low_mean', 'low_std', 'volume_mean', 'volume_std']]
        y = df['close']

        # Impute missing values in features
        X_imputed = imputer.fit_transform(X)

        # Initialize the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model using the imputed data
        model.fit(X_imputed, y)

        # Get the last available row for the current stock
        latest_stock_data = df.tail(1)

        # Create lag features for the day you want to predict
        #latest_stock_data['lag_close_1'] = latest_stock_data['close'].shift(1)
        latest_stock_data.loc[:, 'lag_close_1'] = latest_stock_data['close'].shift(1)


        # Extract features for prediction
        stock_features = latest_stock_data[['day', 'month', 'day_of_week', 'lag_close_1', 'close_mean', 'close_std', 'open_mean', 'open_std', 'high_mean', 'high_std', 'low_mean', 'low_std', 'volume_mean', 'volume_std']].values

        # Impute missing values for prediction
        stock_features_imputed = imputer.transform(stock_features)

        # Make prediction using the model and imputed features
        stock_prediction = model.predict(stock_features_imputed)
        #print("iterator: " + str(iterator))
        #print("start: " + str(currentDate))
        #print("end: " + str(endDate))
        #print("REC DATE: " + str(RecieverDate))

        storer.append(f'The next stock price for {stock_ticker} will be: {stock_prediction[0]}')
        cursor = conn.cursor()
        updatePredictionquery = '''
            UPDATE ALLONE
            SET prediction = ?
            WHERE name = ? AND date = ?'''
        cursor.execute(updatePredictionquery, (stock_prediction[0], stock_ticker, RecieverDate))
        conn.commit()
        #print("yuh")


    #end of for loop, time to iterate
    iterator += 1
    currentDate += timedelta(days=1)
    endDate += timedelta(days=1)
    RecieverDate += timedelta(days=1)

# Close the database connection
conn.close()

print("\n")
for i in storer:
    print(i)
