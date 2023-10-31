import sqlite3
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor



def predict_stock_prices(hidden_size, num_layers, sequence_length, num_epochs, num_lag_features):
    # Connect to the SQLite database
    conn = sqlite3.connect('eltee.db')

    # List of stock tickers
    stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']
    storer = {}

    # Initialize the imputer
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()

    # Extract features from the DataFrame
    features = ['open', 'high', 'low', 'close', 'volume']

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
        features = ['open', 'high', 'low', 'close', 'volume']

        for i in range(1, num_lag_features + 1):
            col_name = f'lag_close_{i}'
            df[col_name] = df['close'].shift(i)
            features.append(col_name)

        input_size = len(features)

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
        #storer.append(f'The next stock price for {stock_ticker} will be: {predicted_features[3]}')  # Close price is at index 3
        #print(f'The next stock price for {stock_ticker} will be: {predicted_features[3]}')  # Close price is at index 3)
        storer[stock_ticker] = predicted_features[3]

    # Close the database connection
    conn.close()

    return storer



#___________________________________________________________________________________________________________________


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#MAIN FILE
#bruh = predict_stock_prices(40,3,25,50,2)
#print(bruh)
#print(type(bruh))
#exit()

#THE HYPERPARAMETER ORDER IS 
#hidden_size, num_layers, sequence_length, num_epochs, num_lag_features
#THE MAIN IDEA IS TO MAKE A BUNCH OF LTSM MODELS AND LET RANDOMFORREST TUNE MY STUFF 
hyperparameters = [
    [50, 3, 25, 100, 8],
    [50, 3, 30, 100, 3],
    [50, 2, 30, 100, 3],
    [22, 2, 10, 150, 6],
    [70, 12, 16, 150, 15],
    [40, 2, 25, 90, 7],
    [50, 1, 15, 100, 7],
    [50, 2, 11, 150, 5],
    [32, 1, 15, 50, 4],
    [30, 1, 12, 45, 4],
    [30, 1, 11, 40, 3],
    [28, 2, 11, 100, 3],
    [33, 2, 17, 55, 5],
    [34, 2, 18, 57, 5],
    [35, 2, 19, 58, 5],
    [37, 3, 21, 58, 6]
]

#this is j shorter for testing 
hyperparameters = [
    [10, 3, 25, 100, 8],
    [10, 3, 30, 100, 3],
    [10, 2, 30, 100, 3]
]
#we could make this like a crazy dict but honestly im too damn lazy to think hahahah!!!!
hidden_size = 0
num_layers = 1
sequence_length = 2
num_epochs = 3
num_lag_features = 4
stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']
iterator = 0
df = pd.DataFrame()
for row in hyperparameters:
    iterator += 1
    df["MODEL " + str(iterator)] = predict_stock_prices(row[0], row[1], row[2], row[3], row[4])
    #I swear we're close we just need to random forrest this bad boy 
    #we're getting this so far
#             MODEL 1     MODEL 2     MODEL 3
#RTX    75.539215   76.011543   76.424179
#NOC   426.148865  488.924988  429.696350
#GD    242.797424  236.387543  240.245316
#LDOS   93.501678   93.352325   93.175217
#KBR    59.744396   59.669979   59.443161
#BWXT   77.069778   77.081123   76.638786
#RKLB    4.279412    4.275094    4.232845
#LMT   447.178619  436.375916  449.385651

print(df)

exit()

stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']
iterator = 0
df = pd.DataFrame()
for row in hyperparameters:
    iterator += 1
    stockDict = {}
    for stock_ticker in stocklist:
        stockDict[stock_ticker] = predict_stock_prices(row[0], row[1], row[2], row[3], row[4])


    df["MODEL " + str(iterator)] = [1,2,3,4]

print(df)