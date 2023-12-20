import requests
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date

api_key = "3127cbac6af39b2f5fbab8b2caf075be"

# Define the endpoint URL for intraday stock prices
endpoint_url = 'https://www.alphavantage.co/query'

# Define query parameters
params = {
    'function': 'TIME_SERIES_INTRADAY',
    'symbol': 'AAPL',
    'interval': '5min',  # Adjust the interval as needed (e.g., 1min, 15min)
    'apikey': api_key,
}

# Make the API request
response = requests.get(endpoint_url, params=params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse and print the response JSON data
    data = response.json()
    print(data)
else:
    # Print an error message if the request was not successful
    print(f"Error: {response.status_code} - {response.text}")

print("okafor")

'''
so the idea is 
have the date
have the prev day's closing price (maybe as a special variable)
then 5 min intervals throughout the whole day 
I want the open and close and volume'''