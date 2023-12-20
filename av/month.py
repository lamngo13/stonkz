import requests
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date

api_key = "3127cbac6af39b2f5fbab8b2caf075be"

# Construct the API request URL for intraday data for January 2009 for IBM
endpoint_url = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_INTRADAY',
    'symbol': 'APPL',
    'interval': '5min',
    'month': '2023-11',
    'outputsize': 'full',
    'apikey': api_key,
}

# Make the API request
response = requests.get(endpoint_url, params=params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse and print the response JSON data
    data = response.json()

    # Extract and print relevant information
    intraday_data = data.get('Time Series (5min)', {})
    for timestamp, values in intraday_data.items():
        open_price = values.get('1. open', 'N/A')
        close_price = values.get('4. close', 'N/A')
        volume = values.get('5. volume', 'N/A')

        print(f'Timestamp: {timestamp}, Open: {open_price}, Close: {close_price}, Volume: {volume}')
else:
    # Print an error message if the request was not successful
    print(f"Error: {response.status_code} - {response.text}")

