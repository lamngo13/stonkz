import requests
import json
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date

# Replace '20230105' with the desired date in the format YYYYMMDD
target_date = '2023-12-01'
def get_intraday(date: str):
    print("FROM THE RIVER TO THE SEA")
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

def create_table(input_count=None):
    print("starting create_table function - also free Palestine")
    if input_count is None:
        #init to 170, else read the file
        count = 158
        #this gets about like 11:40ish at night, the full 170 goes into the next day 
    else:
        #check for the true count of the elements only if we have an argument
        file_path = 'a.txt'
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            fc = json.load(file)

        count = fc['resultsCount']

    #print("COUNT: " + str(count))

    #this line should remain unchanged, but the 'count' should change conditionally 
    columns = ', '.join([f'open{i} TEXT, close{i} TEXT, high{i} TEXT, time{i} TEXT' for i in range(count)])
    #create the table based on the number of columns
    create_table_query = f'''
        CREATE TABLE IF NOT EXISTS appl (
            id INTEGER PRIMARY KEY,
            date TEXT,
            prevclose TEXT,
            {columns}
        )
    '''
    print(create_table_query)
    #ok now actually create the database
    conn = sqlite3.connect('appl.db')
    cursor = conn.cursor()
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()

def unix_to_real(unix: int):
    seconds = unix / 1000
    dt_object = datetime.utcfromtimestamp(seconds)
    print(dt_object)
    twenty_four = dt_object.strftime('%H:%M:%S')
    print(twenty_four)
    return twenty_four


def parsing(date:str):
    print("FROM THE RIVER TO THE SEA")
    holder = get_intraday(date)
    #this contains the actual api call

    count = int(holder['resultsCount'])
    conn = sqlite3.connect('river.db')
    # Create a cursor
    cursor = conn.cursor()

    # Create a table
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS river (
        id INTEGER PRIMARY KEY,
        date, TEXT
        name, TEXT,
        open TEXT,
        high TEXT,
        low TEXT,
        close TEXT,
        volume TEXT
        )
        '''