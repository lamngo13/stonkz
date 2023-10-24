#howdy
#howdy
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#stocklist
stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']

# Connect to the SQLite database
conn = sqlite3.connect('ALLONE.db')

# Query to retrieve stock data for a specific date range
query = '''
    SELECT date, open, high, low, close, volume, name 
    FROM ALLONE 
    WHERE date BETWEEN '2023-10-01' AND '2023-10-20'
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

# Create lag features for the previous day's closing price
df['lag_close_1'] = df.groupby('name')['close'].shift(1)

# Calculate rolling mean and rolling standard deviation
window = 3
df['rolling_mean'] = df.groupby('name')['close'].rolling(window=window).mean().reset_index(0, drop=True)
df['rolling_std'] = df.groupby('name')['close'].rolling(window=window).std().reset_index(0, drop=True)

# One-hot encode stock names
df = pd.get_dummies(df, columns=['name'], prefix='stock')

# Drop NaN values
df = df.dropna()

# Split data into features (X) and target variable (y)
X = df[['day', 'month', 'day_of_week', 'lag_close_1', 'rolling_mean', 'rolling_std', 'stock_RTX', 'stock_NOC', 'stock_GD','stock_LDOS', 'stock_KBR', 'stock_BWXT', 'stock_RKLB', 'stock_LMT']]
y = df['close']

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Impute missing values in training and testing features
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the imputed training data
model.fit(X_train_imputed, y_train)

# Make predictions on the imputed test data
predictions = model.predict(X_test_imputed)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Predict specific stock ('LMT')
# Get the last available row for 'LMT'
latest_lmt_data = df[df['stock_LMT'] == 1].tail(1)

# Create lag features for the day you want to predict
latest_lmt_data['lag_close_1'] = latest_lmt_data['close'].shift(1)

# Extract features for prediction
LMT_features = latest_lmt_data[['day', 'month', 'day_of_week', 'lag_close_1', 'rolling_mean', 'rolling_std']].values

# Impute missing values for prediction
LMT_features_imputed = imputer.transform(LMT_features)

# Make prediction using the model and imputed features
LMT_prediction = model.predict(LMT_features_imputed)

print(f'The next stock price for LMT will be: {LMT_prediction[0]}')
