import requests
import json

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
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
    # Save the response to a text file
    with open('text.txt', 'w') as file:
        file.write(json.dumps(response.json(), indent=4))

    print("Data saved to intraday_data_response.txt.")
else:
    # Print an error message if the request was not successful
    print(f"Error: {response.status_code} - {response.text}")
