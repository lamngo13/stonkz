#howdy
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('ALLONE.db')

# Load data from the database into a pandas DataFrame
query = 'SELECT date, open, high, low, close, volume FROM stock_data WHERE stockname = "AAPL"'
df = pd.read_sql(query, conn)

# Close the database connection
conn.close()
