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
    start_time = datetime.strptime("09:00", "%H:%M")
    iterator_values = range(count)
    time_values = [(start_time + timedelta(minutes=i * 5)).strftime("%H%M") for i in iterator_values]

    columns = ', '.join([f'open_{i} TEXT, close_{i} TEXT, high_{i} TEXT, low_{i} TEXT, volume_{i} TEXT, N_{i} TEXT, unix_time_{i} TEXT' for i in time_values])
    #create the table based on the number of columns
    create_table_query = f'''
        CREATE TABLE IF NOT EXISTS appl1 (
            id INTEGER PRIMARY KEY,
            date TEXT,
            prev_close TEXT,
            {columns}
        )
    '''
    print(create_table_query)
    #ok now actually create the database
    conn = sqlite3.connect('appl1.db')
    cursor = conn.cursor()
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()

def unix_to_real(unix: int):
    seconds = unix / 1000
    dt_object = datetime.utcfromtimestamp(seconds)
    twenty_four = dt_object.strftime('%H:%M')
    print(twenty_four)
    return twenty_four

def first_row():
    file_path = 'a.txt'
        # Open the file and read its contents
    with open(file_path, 'r') as file:
        fc = json.load(file)

    res = fc['results']
    conn = sqlite3.connect("appl1.db")
    cursor = conn.cursor()

    columns_to_insert = []
    start_time = datetime.strptime("09:00", "%H:%M")
    end_time = datetime.strptime("22:05", "%H:%M")
    interval = timedelta(minutes=5)
    current_time = start_time
    iterator_values = []
    while current_time <= end_time:
        iterator_values.append(current_time.strftime("%H%M"))
        current_time += interval

    # append to using columns
    for value in iterator_values:
        columns_to_insert.append("open_"+str(value))
        columns_to_insert.append("close_"+str(value))
        columns_to_insert.append("high_"+str(value))
        columns_to_insert.append("low_"+str(value))
        columns_to_insert.append("volume_"+str(value))
        columns_to_insert.append("N_"+str(value))
        columns_to_insert.append("unix_time_"+str(value))

    values = []
    for row in res:
        values.append(row['o'])
        values.append(row['c'])
        values.append(row['h'])
        values.append(row['l'])
        values.append(row['v'])
        values.append(row['n'])
        values.append(row['t'])

    values = [str(value) for value in values]  # Convert values to strings
    #cut down the values in our dataset
    while (len(columns_to_insert) < len(values)):
        values.pop()

    '''
    print(values)
    print(columns_to_insert)
    print(len(values))
    print(len(columns_to_insert))
    exit()
    '''

    # Construct the INSERT query
    insert_query = f"INSERT INTO appl1 ({', '.join(columns_to_insert)}) VALUES ({', '.join(values)})"

    # Execute the INSERT query
    cursor.execute(insert_query)


    conn.commit()
    conn.close()

'''
# Sample data
values_to_insert = [1, 2, 3, 4, 5]

# Construct the INSERT query
insert_query = f"INSERT INTO your_table_name ({', '.join(['bruh' + str(i) for i in range(1, len(values_to_insert) + 1)])}) VALUES ({', '.join(['?' for _ in values_to_insert])})"
'''

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