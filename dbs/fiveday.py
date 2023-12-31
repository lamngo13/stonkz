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
        WHERE date BETWEEN '2023-09-24' AND '2023-10-20' AND name = '{stock_ticker}'
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
    df['lag_close_1'] = df['close'].shift(5)  # Assuming 5 trading days in a week

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
    latest_stock_data['lag_close_1'] = latest_stock_data['close'].shift(5)  # Assuming 5 trading days in a week

    # Extract features for prediction
    stock_features = latest_stock_data[['day', 'month', 'day_of_week', 'lag_close_1', 
                                       'rolling_mean', 'rolling_std']].values

    # Impute missing values for prediction
    stock_features_imputed = imputer.transform(stock_features)

    # Make prediction using the model and imputed features
    stock_prediction = model.predict(stock_features_imputed)

    storer.append(f'The next stock price for {stock_ticker} will be: {stock_prediction[0]}')

# Close the database connection
conn.close()

print("\n")
for i in storer:
    print(i)
