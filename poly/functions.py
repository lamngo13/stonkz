import requests
import json
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date
import time
import os

# Replace '20230105' with the desired date in the format YYYYMMDD
#target_date = '2023-12-01'
def get_intraday(date: str, filename: str):
    print("FROM THE RIVER TO THE SEA")
    #YYYYMMDD
    #ex '2023-12-01'
    api_key = "TRxer9Mhmo64ERvyE5mRbrQI69Atdo7v"
    symbol = 'AAPL'

    #gets intraday data for a single given day
    endpoint_url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{date}/{date}?unadjusted=false&apiKey={api_key}'

    # Make the API request
    response = requests.get(endpoint_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the response JSON data
        data = response.json()
        print(data)

        # Write the response to a text file named 'a.txt'
        with open(f'{filename}.txt', 'w') as file:
            file.write(json.dumps(data, indent=4))

        print(f"Data saved to {filename}.txt.")
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


    insert_query = f"INSERT INTO appl1 ({', '.join(columns_to_insert)}) VALUES ({', '.join(values)})"
    cursor.execute(insert_query)
    conn.commit()
    conn.close()

def mass_txt_db():
    #TODO TODO TODO 
    #we need 07-03 to 11-30 inclusive
    start_date = datetime(2023, 7, 3)
    start_date = datetime(2023, 7, 2)
    #jan 1 and 2 were weekends
    end_date = datetime(2023, 11, 30) #make this like 10 j for jan for testing
    start_date = datetime(2023, 7, 3)

    # Define the step (1 day in this case)
    step = timedelta(days=1)

    # Loop through dates
    current_date = start_date
    while current_date <= end_date:
        date_string = current_date.strftime("%Y-%m-%d")
        file_path = f"{date_string}.txt"

        if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
            print("trying for " + str(date_string))
            into_db(date_string)
        else:
            print(f"The file {file_path} does not exist or is empty.")

        #while condition, end while loop after
        current_date += step

        

def into_db(file_name):
    file_path = f'{file_name}.txt'
        # Open the file and read its contents
    with open(file_path, 'r') as file:
        fc = json.load(file)

    res = fc['results']
    conn = sqlite3.connect("appl1.db")
    cursor = conn.cursor()

    columns_to_insert = []
    values = []

    start_time = datetime.strptime("09:00", "%H:%M")
    end_time = datetime.strptime("22:05", "%H:%M")
    interval = timedelta(minutes=5)
    current_time = start_time
    iterator_values = []
    while current_time <= end_time:
        iterator_values.append(current_time.strftime("%H%M"))
        current_time += interval

    #parsed_date = datetime.strptime(file_name, "%Y-%m-%d")
    #formatted_date = parsed_date.strftime("%m/%d/%Y")
    #formatted_date = file_name.strftime("%m/%d/%Y")
    #columns_to_insert.append("date")
    #values.append(str(formatted_date))

    # append to using columns
    for value in iterator_values:
        columns_to_insert.append("open_"+str(value))
        columns_to_insert.append("close_"+str(value))
        columns_to_insert.append("high_"+str(value))
        columns_to_insert.append("low_"+str(value))
        columns_to_insert.append("volume_"+str(value))
        columns_to_insert.append("N_"+str(value))
        columns_to_insert.append("unix_time_"+str(value))

    for row in res:
        values.append(row['o'])
        values.append(row['c'])
        values.append(row['h'])
        values.append(row['l'])
        values.append(row['v'])
        values.append(row['n'])
        values.append(row['t'])

    
    print("lencol: " + str(len(columns_to_insert)))
    print("lenvals: " + str(len(values)))

    values = [str(value) for value in values]  # Convert values to strings
    #cut down the values in our dataset
    while (len(columns_to_insert) < len(values)):
        values.pop()

    print("post pop:")
    print("lencol: " + str(len(columns_to_insert)))
    print("lenvals: " + str(len(values)))

    #hardcode date into here
    '''
    parsed_date = datetime.strptime(file_name, "%Y-%m-%d")
    formatted_date = parsed_date.strftime("%m/%d/%Y")
    columns_to_insert.append("date")
    values.append(str(formatted_date))
    '''
    #print(len(columns_to_insert))
    #print(len(values))
    #print(columns_to_insert)
    #print(values)
    #exit()

    insert_query = f"INSERT INTO appl1 ({', '.join(columns_to_insert)}) VALUES ({', '.join(values)})"
    #print(insert_query)
    #exit()
    cursor.execute(insert_query)
    conn.commit()
    conn.close()

def hardcode_first():
    #prev close is 189.95
    conn = sqlite3.connect("appl1.db")
    cursor = conn.cursor()
    prev_close = "189.95"
    the_date = datetime(2023, 12, 1)
    the_date = the_date.strftime("%m/%d/%Y")
    the_date = f"'{the_date}'"
    insert_query = f"UPDATE appl1 SET date = {the_date}, prev_close = {prev_close} WHERE id = 1"
    cursor.execute(insert_query)
    conn.commit()
    conn.close()

def prev(id: int, year: int, month: int, day: int):
    #get date from args
    the_date = datetime(year, month, day)
    the_date = the_date.strftime("%m/%d/%Y")
    the_date = f"'{the_date}'"

    #get previous id and also current id
    prev_id = id - 1
    id = f"'{id}'"
    
    conn = sqlite3.connect("appl1.db")
    cursor = conn.cursor()
    prev_query_string = f"SELECT close_2205 FROM appl1 WHERE id = {prev_id}"
    cursor.execute(prev_query_string)
    result = cursor.fetchone()
    prev_close_string = str(result[0])
    prev_close_string = f"'{prev_close_string}'"
    insert_query = f"UPDATE appl1 SET date = {the_date}, prev_close = {prev_close_string} WHERE id = {id}"
    cursor.execute(insert_query)
    conn.commit()
    conn.close()

def orderer():
    print("sorting by unix_time_0900")
    conn = sqlite3.connect('appl1.db')
    cursor = conn.cursor()

    # Query to reorder the table by date in ascending order and update the original table
    query = '''
        CREATE TABLE temp_table AS
        SELECT * FROM appl1
        ORDER BY DATE(unix_time_0900) ASC;
        
        DROP TABLE appl1;
        
        ALTER TABLE temp_table RENAME TO appl1;
    '''

    # Execute the query
    cursor.executescript(query)
    conn.commit()

    # Fetch the results if needed
    cursor.execute("SELECT * FROM ALLONE")
    results = cursor.fetchall()

    # Process the results as needed
    for row in results:
        print(row)

    # Close the connection
    conn.close()

def date_adder():
    #16 to 141
    for i in range(16, 142):
        conn = sqlite3.connect("appl1.db")
        cursor = conn.cursor()
        prev_query_string = f"SELECT unix_time_0905 FROM appl1 WHERE id = {i}"
        cursor.execute(prev_query_string)
        result = cursor.fetchone()
        result = int(int(str(result[0]))/1000)
        dt_object = datetime.utcfromtimestamp(result)
        date_string = dt_object.strftime("%m-%d-%Y")
        update_query = f"UPDATE appl1 SET date = {the_date}, prev_close = {prev_close_string} WHERE id = {id}"

        cursor.execute(update_query)
        conn.commit()
        conn.close()

    exit()


    #get date from args
    the_date = datetime(year, month, day)
    the_date = the_date.strftime("%m/%d/%Y")
    the_date = f"'{the_date}'"

    #get previous id and also current id
    prev_id = id - 1
    id = f"'{id}'"
    
    conn = sqlite3.connect("appl1.db")
    cursor = conn.cursor()
    prev_query_string = f"SELECT close_2205 FROM appl1 WHERE id = {prev_id}"
    cursor.execute(prev_query_string)
    result = cursor.fetchone()
    prev_close_string = str(result[0])
    prev_close_string = f"'{prev_close_string}'"
    insert_query = f"UPDATE appl1 SET date = {the_date}, prev_close = {prev_close_string} WHERE id = {id}"
    cursor.execute(insert_query)
    conn.commit()
    conn.close()


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
    
#starting at jan 1 2023
def autofiller():
    #USE THIS TO MASS FILL TEXT FILES WITH STOCK DATA
    #this is a counter for my limited api key stuff
    counter = 1
    api_key = "TRxer9Mhmo64ERvyE5mRbrQI69Atdo7v"
    symbol = 'AAPL'

    start_date = datetime(2023, 7, 29)
    end_date = datetime(2023, 11, 30)

    current_date = start_date

    while current_date <= end_date:

        #idle if reaching api limit
        if ((counter % 5) == 0):
            time.sleep(61)
        #formate date for api
        formatted_current_date = current_date.strftime('%Y-%m-%d')

        #create file name
        filename = str(formatted_current_date)
        #MAKE THE CALL HERE
        endpoint_url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{formatted_current_date}/{formatted_current_date}?unadjusted=false&apiKey={api_key}'

        # Make the API request
        response = requests.get(endpoint_url)

        # Check if the request was successful 
        if response.status_code == 200:
            # Parse the response JSON data
            data = response.json()
            if (data['resultsCount']):
                if (int(data['resultsCount']) > 50):
                    #do the stuff
                    print("writing for date: " + str(formatted_current_date))
                    # Write the response to a text file named 'a.txt'
                    with open(f'{filename}.txt', 'w') as file:
                        file.write(json.dumps(data, indent=4))

                    print(f"Data saved to {filename}.txt.")
                    #IF THIS IS THE CASE THEN WRITE TO DB!!!!
                else:
                    print("result count is < 50 for " + str(formatted_current_date))
            else:
                print("resultCount does not exist! for " + str(formatted_current_date))
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")

            
        #while condition
        #should be inside while loop btw
        current_date += timedelta(days=1)
        #increment counter for api limit
        counter = counter + 1

    print("DONZO")


def hard_delete(zid):
    id = str(zid)
    # Connect to the SQLite database
    conn = sqlite3.connect('appl1.db')  # Replace 'your_database.db' with your actual database file

    # Create a cursor
    cursor = conn.cursor()

    try:
        # Define the DELETE query
        delete_query = f"DELETE FROM appl1 WHERE id = ?"

        # Execute the query with the provided ID
        cursor.execute(delete_query, (id,))

        # Commit the changes to the database
        conn.commit()

        print(f"Row with ID {id} deleted successfully.")

    except sqlite3.Error as e:
        print(f"Error deleting row: {e}")

    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()