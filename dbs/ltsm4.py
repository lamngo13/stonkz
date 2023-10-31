import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Connect to the SQLite database
conn = sqlite3.connect('eltee.db')

# List of stock tickers
stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']
storer = []

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()

# Extract features from the DataFrame
features = ['open', 'high', 'low', 'close', 'volume']

# Training parameters
#COOL HYPERPARAMATERS THIS IS PROB THE FUN PART HEHE
hidden_size = 50
#idk?
num_layers = 5
#should prob be as high as tolerable but 5 seems okay
sequence_length = 30
#should prob be as high as tolerable but who knows
num_epochs = 100
#TRULY HAVE NO IDEA
#THIS will be a new hyperparameter to hardcode previous number of days as features 
input_size = len(features) #TODO maybe change this to include num_prev_days

num_lag_features = 7
#THIS will be a new hyperparameter to hardcode previous number of days as features 


# Loop through each stock ticker
for stock_ticker in stocklist:
    features = ['open', 'high', 'low', 'close', 'volume']
    # Query to retrieve stock data for the current stock ticker within a specific date range
    query = f'''
        SELECT date, open, high, low, close, volume
        FROM ALLONE 
        WHERE name = '{stock_ticker}'
    '''
    # Load data from the database into a pandas DataFrame
    df = pd.read_sql(query, conn)

    # Extract date features
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek



    features = ['open', 'high', 'low', 'close', 'volume']

    for i in range(1, num_lag_features + 1):
        col_name = f'lag_close_{i}'
        df[col_name] = df['close'].shift(i)
        features.append(col_name)

    input_size = len(features)
    print(len(df))
    #print(df)
    print("bruh")
    for col in df.columns:
        print(col)
    print(features)
    print("YEET: " + str(len(features)))
    #exit()
    # Calculate rolling mean and rolling standard deviation
    #window = 3
    #df['close_mean'] = df['close'].rolling(window=window).mean()
    #df['close_std'] = df['close'].rolling(window=window).std()

    # Drop NaN values
    df = df.dropna()

    # Extract necessary features
    X = df[features].values

    # Normalize the data
    X = scaler.fit_transform(X)

    # Create sequences of data for LSTM training
    X_sequences, y = [], []
    for i in range(sequence_length, len(X)):
        X_sequences.append(X[i-sequence_length:i])
        y.append(X[i])  # Include all features in y, not just the close price

    X_sequences, y = np.array(X_sequences), np.array(y)

    # Convert data to PyTorch tensors
    X_sequences = torch.tensor(X_sequences, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Define the LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size)  # Output all features

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])  # Get output from the last time step
            return output

    # Initialize the LSTM model
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training the LSTM model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_sequences)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Making predictions
    model.eval()
    with torch.no_grad():
        # Extract the last sequence for each stock ticker
        X_last_sequence = torch.tensor(X[-sequence_length:].reshape(1, sequence_length, -1), dtype=torch.float32)
        predictions = model(X_last_sequence)
        predicted_features = scaler.inverse_transform(predictions.numpy())[0]

    # Update the database with the predicted values

    storer.append(f'The next stock price for {stock_ticker} will be: {predicted_features[3]}')  # Close price is at index 3
    print(f'The next stock price for {stock_ticker} will be: {predicted_features[3]}')  # Close price is at index 3)

# Close the database connection
conn.close()

print("\n")
for i in storer:
    print(i)
