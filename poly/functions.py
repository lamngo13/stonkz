import requests
import json

# Replace '20230105' with the desired date in the format YYYYMMDD
target_date = '2023-12-01'
def get_intraday(date: str):
    print("del rio a mar")
    #YYYYMMDD
    #ex '2023-12-01'
    api_key = "TRxer9Mhmo64ERvyE5mRbrQI69Atdo7v"
    symbol = 'AAPL'

    #gets intraday data for a single given day
    endpoint_url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{target_date}/{target_date}?unadjusted=false&apiKey={api_key}'

    # Make the API request
    response = requests.get(endpoint_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the response JSON data
        data = response.json()
        print(data)

        # Write the response to a text file named 'a.txt'
        with open('a.txt', 'w') as file:
            file.write(json.dumps(data, indent=4))

        print("Data saved to a.txt.")
        return data
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code} - {response.text}")



target_date = '2023-12-01'

get_intraday(target_date)
