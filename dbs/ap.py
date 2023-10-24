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
        WHERE date BETWEEN '2023-10-01' AND '2023-10-20' AND name = '{stock_ticker}'
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
    #TODO why are we doing this?
    window = 3

    #close
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



    # Drop NaN values
    df = df.dropna()

    print("yuh")
    print(df)

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
    latest_stock_data['lag_close_1'] = latest_stock_data['close'].shift(1)

    # Extract features for prediction
    stock_features = latest_stock_data[['day', 'month', 'day_of_week', 'lag_close_1', 'close_mean', 'close_std', 'open_mean', 'open_std', 'high_mean', 'high_std', 'low_mean', 'low_std', 'volume_mean', 'volume_std']].values

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

#assuming for 2023-10-23
#The next stock price for RTX will be: 73.06670181274414
#The next stock price for NOC will be: 485.91860931396485 !!!!!!!!!!!!!!!!
#The next stock price for GD will be: 236.46566650390625
#The next stock price for LDOS will be: 93.00990005493163 !!!!!!!!!!!!!!!!!!!!!!!
#The next stock price for KBR will be: 59.38069984436035
#The next stock price for BWXT will be: 76.84980087280273
#The next stock price for RKLB will be: 4.281900053024292
#The next stock price for LMT will be: 443.78630584716797

#assuming for 2023-10-23 with more features
#The next stock price for RTX will be: 73.02210166931152
#The next stock price for NOC will be: 478.9236083984375
#The next stock price for GD will be: 235.83083068847657
#The next stock price for LDOS will be: 93.01390022277832
#The next stock price for KBR will be: 59.46889976501465
#The next stock price for BWXT will be: 76.81240135192871
#The next stock price for RKLB will be: 4.249200053215027
#The next stock price for LMT will be: 443.26680847167967