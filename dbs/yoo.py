# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer

# Connect to the SQLite database
conn = sqlite3.connect('ALLONE.db')

# Query to retrieve stock data for 'LMT' for a specific date range
query = '''
    SELECT date, open, high, low, close, volume 
    FROM ALLONE 
    WHERE date BETWEEN '2023-10-01' AND '2023-10-20' AND name = 'LMT'
'''

# Load data from the database into a pandas DataFrame
df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

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

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Impute missing values in features
X_imputed = imputer.fit_transform(X)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the imputed data
model.fit(X_imputed, y)

# Predict specific stock ('LMT')
# Get the last available row for 'LMT'
latest_lmt_data = df.tail(1)

# Create lag features for the day you want to predict
latest_lmt_data['lag_close_1'] = latest_lmt_data['close'].shift(1)

# Extract features for prediction
LMT_features = latest_lmt_data[['day', 'month', 'day_of_week', 'lag_close_1', 
                                'rolling_mean', 'rolling_std']].values

# Impute missing values for prediction
LMT_features_imputed = imputer.transform(LMT_features)

# Make prediction using the model and imputed features
LMT_prediction = model.predict(LMT_features_imputed)

print(f'The next stock price for LMT will be: {LMT_prediction[0]}')
