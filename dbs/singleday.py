# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer

# List of stock tickers
stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']
storer = []

# Connect to the SQLite database
conn = sqlite3.connect('ALLONE.db')

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Loop through each stock ticker
for stock_ticker in stocklist:
    # Query to retrieve stock data for the current stock ticker within a specific date range
    query = f'''
        SELECT date, open, high, low, close, volume 
        FROM ALLONE 
        WHERE date BETWEEN '2023-10-01' AND '2023-10-27' AND name = '{stock_ticker}'
    '''

    # Load data from the database into a pandas DataFrame
    df = pd.read_sql(query, conn)

    #---------------------END RETRIEVAL FROM DB----------------------------
    # Extract date features
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Create lag features for previous day's closing price
    df['lag_close_1'] = df['close'].shift(1)

    # Calculate rolling mean and rolling standard deviation
    window = 3
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()

    # Drop NaN values
    df = df.dropna()

    # Split data into features (X) and target variable (y)
    X = df[['day', 'month', 'day_of_week', 'lag_close_1', 'rolling_mean', 'rolling_std']]
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
    latest_stock_data['lag_close_1'] = latest_stock_data['close'].shift(1)

    # Extract features for prediction
    stock_features = latest_stock_data[['day', 'month', 'day_of_week', 'lag_close_1', 
                                       'rolling_mean', 'rolling_std']].values

    # Impute missing values for prediction
    stock_features_imputed = imputer.transform(stock_features)

    # Make prediction using the model and imputed features
    stock_prediction = model.predict(stock_features_imputed)

    #print(f'The next stock price for {stock_ticker} will be: {stock_prediction[0]}')
    storer.append(f'The next stock price for {stock_ticker} will be: {stock_prediction[0]}')

# Close the database connection
conn.close()

print("\n")
for i in storer:
    print(i)

#from ltsm
#The next stock price for RTX will be: 77.38436889648438
#The next stock price for NOC will be: 474.0264587402344
#The next stock price for GD will be: 238.8101348876953
#The next stock price for LDOS will be: 92.97588348388672
#The next stock price for KBR will be: 59.88546371459961
#The next stock price for BWXT will be: 75.37986755371094
#The next stock price for RKLB will be: 4.702893257141113
#The next stock price for LMT will be: 434.6736755371094

#ltsm with more layers
#The next stock price for RTX will be: 77.83975982666016
#The next stock price for NOC will be: 457.97308349609375
#The next stock price for GD will be: 227.5014190673828
#The next stock price for LDOS will be: 93.96693420410156
#The next stock price for KBR will be: 60.027557373046875
#The next stock price for BWXT will be: 75.29927062988281
#The next stock price for RKLB will be: 4.160778522491455
#The next stock price for LMT will be: 429.79718017578125

#ltsm2 with more layers and greater sequence
#The next stock price for RTX will be: 77.50206756591797
#The next stock price for NOC will be: 462.56683349609375
#The next stock price for GD will be: 224.27565002441406
#The next stock price for LDOS will be: 91.66226959228516
#The next stock price for KBR will be: 58.759456634521484
#The next stock price for BWXT will be: 74.57535552978516
#The next stock price for RKLB will be: 4.272033214569092
#The next stock price for LMT will be: 438.1769104003906
    
#from singleday
#The next stock price for RTX will be: 77.70670181274414
#The next stock price for NOC will be: 478.53520141601564
#The next stock price for GD will be: 239.02880081176758
#The next stock price for LDOS will be: 92.03229957580567
#The next stock price for KBR will be: 58.99599975585937
#The next stock price for BWXT will be: 75.64330085754395
#The next stock price for RKLB will be: 4.289700074195862
#The next stock price for LMT will be: 444.13280303955077