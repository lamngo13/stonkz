import requests
import json
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date
import time
import os
import pandas as pd

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
    #missing 07/03/2023
    #and missing 11/24/2023
    start_date = datetime(2023, 7, 3)
    start_date = datetime(2023, 11, 25)
    end_date = datetime(2023, 11, 30) #make this like 10 j for jan for testing

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
    print("sorting by unix_time_0905")
    conn = sqlite3.connect('appl1.db')
    cursor = conn.cursor()

    # Query to reorder the table by date in ascending order and update the original table
    query = '''
        CREATE TABLE temp_table AS
        SELECT * FROM appl1
        ORDER BY unix_time_0905 ASC;
        
        DROP TABLE appl1;
        
        ALTER TABLE temp_table RENAME TO appl1;
    '''

    # Execute the query
    cursor.executescript(query)
    conn.commit()
    conn.close()

def better_orderer():
    # Connect to the database
    conn = sqlite3.connect("appl1.db")  # Replace with your actual database file name
    cursor = conn.cursor()

    # Execute the query to select all columns from your_table and order by date
    query = """
    SELECT *
    FROM appl1
    ORDER BY
      SUBSTR(date, 7, 4) || SUBSTR(date, 1, 2) || SUBSTR(date, 4, 2);
    """

    cursor.execute(query)
    conn.commit()
    conn.close()


def date_adder():
    #16 to 141
    for i in range(1, 246):
        print(str(i))
        conn = sqlite3.connect("appl1.db")
        cursor = conn.cursor()
        prev_query_string = f"SELECT unix_time_0905 FROM appl1 WHERE id = {i}"
        cursor.execute(prev_query_string)
        result = cursor.fetchone()
        print("result: " + str(result))
        result = int(int(str(result[0]))/1000)
        dt_object = datetime.utcfromtimestamp(result)
        date_string = dt_object.strftime("%m/%d/%Y")
        date_string = f"'{date_string}'"
        update_query = f"UPDATE appl1 SET date = {date_string} WHERE id = {i}"
        cursor.execute(update_query)
        prev_close_query = f"UPDATE appl1 SET prev_close = 101 WHERE id = {i}"
        cursor.execute(prev_close_query)
        conn.commit()
        conn.close()


    
#starting at jan 1 2023
def autofiller():
    #USE THIS TO MASS FILL TEXT FILES WITH STOCK DATA
    #this is a counter for my limited api key stuff
    counter = 1
    api_key = "TRxer9Mhmo64ERvyE5mRbrQI69Atdo7v"
    symbol = 'AAPL'

    start_date = datetime(2023, 12, 1)
    end_date = datetime(2023, 12, 28)
    #I think this is inclusive

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
    conn = sqlite3.connect('manzana1.db')
    cursor = conn.cursor()
    try:
        delete_query = f"DELETE FROM appl1 WHERE id = ?"
        cursor.execute(delete_query, (id,))
        conn.commit()
        print(f"Row with ID {id} deleted successfully.")
    except sqlite3.Error as e:
        print(f"Error deleting row: {e}")
    finally:
        cursor.close()
        conn.close()



def investigation():
    #TODO TODO TODO 
    #this loops through your text files that do exist, and passes them into 
    #another function, tru_new_db()
    #start at jan 3rd!!!!
    start_date = datetime(2023, 12, 1)
    end_date = datetime(2023, 12, 28) #inclusive i think 
    #THESE ARE INCLUSIVE OF BOTH
    

    # Define the step (1 day in this case)
    step = timedelta(days=1)

    # Loop through dates
    current_date = start_date
    while current_date <= end_date:
        date_string = current_date.strftime("%Y-%m-%d")
        file_path = f"{date_string}.txt"

        if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
            tru_new_db(date_string)
        else:
            print(f"The file {file_path} does not exist or is empty.")

        #while condition, end while loop after
        current_date += step

def tru_new_db(date_string_in):
    print("considering date: " + str(date_string_in))
    conn = sqlite3.connect("manzana1.db")
    cursor = conn.cursor()
    
    columns_to_insert = []
    values = []
    present_list = []
    doerdict = {}

    #get data from text file
    file_path = f"{date_string_in}.txt"
    with open(file_path, 'r') as file:
        fc = json.load(file)
    res = fc['results']
    #btw we are assuming that 
    #for every row, there ARE in fact 8 values
    #if that is not the case,
    #then we are not SQL, but rather SOL

    #grab a sample val to format the date
    temp_date = res[0]['t']
    unix_holder = int(int(temp_date)/1000)
    dt_object = datetime.utcfromtimestamp(unix_holder)
    in_date = dt_object.strftime('%m/%d/%Y')
    in_date = f"{in_date}"

    #insert first few values hardcode
    columns_to_insert.append("date")
    values.append(in_date)
    columns_to_insert.append("prev_close")
    values.append("BAD")

    #see which timestamps (in hours and minutes) are present
    #loop through the res to find that
    #we assume the file exists at this point, and has at least 100 entries
    for row in res:
        #get the time in hrs mins
        unix_holder = int(row['t'])
        unix_holder = unix_holder / 1000
        dt_object = datetime.utcfromtimestamp(unix_holder)
        hours_mins = dt_object.strftime('%H%M')
        present_list.append(hours_mins)
        #TODO ALSO APPEND TO A DICT
        doerdict[hours_mins] = row
 
    #THE PRESENT LIST 
    #is used to show what timestamps DO exist

    #create set of columns to insert, and all of their values
    start_time = datetime.strptime("09:00", "%H:%M")
    end_time = datetime.strptime("20:55", "%H:%M")
    current_time = start_time
    interval = timedelta(minutes=5)
    while current_time <= end_time:
        useable_current_time = current_time.strftime("%H%M")
        time_w_colon = current_time.strftime("%H:%M")
        #append vals to columns_to_insert IN ANY CASE
        #THE VALS ARE VARIABLE
        columns_to_insert.append("open_"+str(useable_current_time))
        columns_to_insert.append("close_"+str(useable_current_time))
        columns_to_insert.append("high_"+str(useable_current_time))
        columns_to_insert.append("low_"+str(useable_current_time))
        columns_to_insert.append("volume_"+str(useable_current_time))
        columns_to_insert.append("N_"+str(useable_current_time))
        columns_to_insert.append("unix_time_"+str(useable_current_time))

        
        if useable_current_time in present_list:
            #add vals to col list
            temp_row = doerdict[useable_current_time]
            values.append(str(temp_row['o']))
            values.append(str(temp_row['c']))
            values.append(str(temp_row['h']))
            values.append(str(temp_row['l']))
            values.append(str(temp_row['v']))
            values.append(str(temp_row['n']))
            values.append(str(temp_row['t']))
        else:
            print("NOT found in: " + useable_current_time)
            #if this is the case, put in dummy values
            for i in range(1,8):
                values.append("BAD")

        #iteration step
        current_time += interval

    quoted_values = [f"'{value}'" if isinstance(value, str) else str(value) for value in values]
    store_values = values
    values = quoted_values

    insert_query = f"INSERT INTO manzana1 ({', '.join(columns_to_insert)}) VALUES ({', '.join(values)})"
    cursor.execute(insert_query)
    conn.commit()
    conn.close()

def fixer_of_bad():
    
    conn = sqlite3.connect("manzana1.db")
    cursor = conn.cursor()
    number_query = "SELECT COUNT(*) FROM manzana1"
    cursor.execute(number_query)
    num_rows = int(cursor.fetchone()[0])

    #fix prev close placeholder of BAD
    for i in range(2,num_rows+1):
        prev_close_query = f"SELECT close_2055 FROM manzana1 WHERE id = {i-1}"
        cursor.execute(prev_close_query)
        prev = str(cursor.fetchone()[0])

        prev_close_updater_query = f'''
        UPDATE manzana1 SET prev_close = {prev} WHERE id = {i}
        '''
        cursor.execute(prev_close_updater_query)

    #next is the tricky part
    #now I need to get each column into a dataframe
    #and use some pandas or numpy function to average out the values that are "BAD"
    #the tricky part is how do I find what I want to put there
       



    conn.commit()
    conn.close()
 
def temp_fixer_of_bad():
    
    conn = sqlite3.connect("manzana1.db")
    table_name = "manzana1"
    cursor = conn.cursor()

    total_db_query = ("SELECT * FROM manzana1")
    df = pd.read_sql_query(total_db_query, conn)
    #print(df.iloc[0])
    #exit()
    #TODO TODO TODO THIS IS FOR A SINGLE DAY!!!!!!

        
    #repeated_cols_list = ['open_', 'close_', 'high_', 'low_', 'volume_', 'N_']
    holder_col = ['open_', 'close_', 'high_', 'low_', 'volume_', 'N_', 'unix_time_']
    holder_col_nounder = ['open', 'close', 'high', 'low', 'volume', 'N', 'unix']
    values = ['1', '2', '3', '4', '5', '6', '7']
    values_dict = {"open": "holder", "close": "holder", "high": "holder", "low": "holder", "volume": "holder", "N": "holder", "unix": "holder"}
    start_time = datetime.strptime("09:00", "%H:%M")
    #start_time = datetime.strptime("09:10", "%H:%M")
    #this for testing
    end_time = datetime.strptime("20:55", "%H:%M")

    #TODO
    #this will be in a larger loop in the future, but we're just gonna hardcode it for now
    this_date = "01/03/2023"
    this_date = 0

    current_time = start_time
    interval = timedelta(minutes=5)
    while current_time <= end_time:

        #just some setup
        useable_current_time = current_time.strftime("%H%M")
        time_w_colon = current_time.strftime("%H:%M")
        holder_col = ['open_', 'close_', 'high_', 'low_', 'volume_', 'N_', 'unix_time_']
        values = ['1', '2', '3', '4', '5', '6', '7']
        values_dict = {"open": "holder", "close": "holder", "high": "holder", "low": "holder", "volume": "holder", "N": "holder", "unix": "holder"}
        
        if (df.iloc[this_date]['open_'+useable_current_time] == "BAD"):
            print(useable_current_time)
            #we only do this stuffski if there is a missing value
            #called "BAD"


            #find the nearest valid previous entry
            #or let it be known that there is none
            closest_prev = "notyet"
            keep_going_back = True
            past_time = current_time
            i = 0
            while(keep_going_back):
                    past_time = past_time - interval
                    formatted_past_time = past_time.strftime("%H%M")
                    #check to see if val for prev time exists
                    try:
                        holder_name = "open_"+formatted_past_time
                        maybe = df[holder_name]
                        #if we here, then ladies n gents, we got em
                        closest_prev = formatted_past_time
                        keep_going_back = False
                        #get outta the shwhile loop
                        #this is weird, but I basically want to keep iterating
                        #and there will be an exception if there is no data there
                    except:
                        dontuse = 420

                    if (i >= 30):
                        keep_going_back = False
                        closest_prev = "impossible"
                    i = i + 1
            #END SHWILE HERE
            
            closest_next = "notyet"
            keep_going_back = True
            past_time = current_time
            i = 0
            while(keep_going_back):
                    past_time = past_time + interval
                    formatted_next_time = past_time.strftime("%H%M")
                    #check to see if val for prev time exists
                    try:
                        holder_name = "open_"+formatted_next_time
                        maybe = df[holder_name]
                        #if we here, then ladies n gents, we got em
                        closest_next = formatted_next_time
                        keep_going_back = False
                        #get outta the shwhile loop
                    except:
                        #this is weird, but I basically want to keep iterating
                        #and there will be an exception if there is no data there
                        dontuse = 420

                    if (i >= 30):
                        keep_going_back = False
                        closest_next = "impossible"
                    i = i + 1
            
            #case of somehow there are no possible replacement vals THIS SHOULD NEVER HAPPEN LMFAOO
            if (closest_next == "impossible" and closest_prev == "impossible"):
                print("NO FREAKING WAY THERE IS NO GOSH DARN DATA!!!!!!!!!!!!!!!!!!")
                print(str(useable_current_time))
                return useable_current_time
            
            
            #case of prev exists but not the next
            if (closest_prev != "impossible" and closest_next == "impossible"):
                print("USING PREV")
                #NOTE
                #it's just easier to overwrite the whole dict and then fix the unix val later
                col_iterator = 0
                for key in values_dict:
                    values_dict[key] = df.iloc[this_date][holder_col[col_iterator]+closest_prev]
                    col_iterator = col_iterator + 1
                #fix unix time

            
            #case of next exists but nor prev
            if (closest_prev == "impossible" and closest_next != "impossible"):
                print("using next alone")
                col_iterator = 0
                for key in values_dict:
                    values_dict[key] = df.iloc[this_date][holder_col[col_iterator]+closest_next]
                    col_iterator = col_iterator + 1
            
            #this will be the most likely case
            if (closest_prev != "impossible" and closest_next != "impossible"):
                #average the two values
                col_iterator = 0
                for key in values_dict:
                    averaged_holder = float(df.iloc[this_date][holder_col[col_iterator]+closest_next]) + float(df.iloc[this_date][holder_col[col_iterator]+closest_prev])
                    averaged_holder = averaged_holder / 2
                    averaged_holder = round(averaged_holder, 2)
                    values_dict[key] = averaged_holder
                    col_iterator = col_iterator + 1
                #replace unix val
                #no change here, bc its the average, just make sure to format later

            #fix unix time
            hours = int(useable_current_time[:2])
            mins = int(useable_current_time[2:])

            #this unix time here is a reference, it is not exact
            #but it should be on the same day and that's all we need!
            ref_timestamp = int(values_dict['unix']) / 1000
            #this should be from the same day
            ref_dt = datetime.utcfromtimestamp(ref_timestamp)
            new_datetime = ref_dt.replace(hour=hours, minute=mins, second=0, microsecond=0)
            new_timestamp = int(new_datetime.timestamp())
            new_timestamp = new_timestamp * 1000
                    
            #TODO FORMAT THE FLOATS AND THEN CONVERT ALL TO STRINGS
            values_dict['unix'] = new_timestamp
            #miliseconds

            #convert all of values_dict to strings
            for z in values_dict:
                values_dict[z] = str(values_dict[z])

            #add this dict back into dataframe
            for i in range(0,len(holder_col)):
                #print("iterator: " + str(i))
                temp_col_name = holder_col[i]+useable_current_time
                df.iloc[this_date][temp_col_name] = values_dict[holder_col_nounder[i]]

            #print(df.iloc[this_date])
            

        
        
        
        #this is the end, lowkey only do the whole loop conditionally
        #this is the end of stuff we do iff BAD

        




        #iteration step
        current_time += interval
       

    #TODO
    #only do this stuff if there is NO, or BAD data here
    #TODO
    #write changes back to the dataframe
    #eventually, loop over the dataframe again with added values
    #until the df is complete and whole
    #then write to the db from the df
    #TODO
    #then loop it for different dates; this is for single day atm
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()





def new_create_table():
    count = 144
    #should be 13 hours
    #0900 - 2055
    start_time = datetime.strptime("09:00", "%H:%M")
    iterator_values = range(count)
    time_values = [(start_time + timedelta(minutes=i * 5)).strftime("%H%M") for i in iterator_values]

    columns = ', '.join([f'open_{i} TEXT, close_{i} TEXT, high_{i} TEXT, low_{i} TEXT, volume_{i} TEXT, N_{i} TEXT, unix_time_{i} TEXT' for i in time_values])
    #create the table based on the number of columns
    create_table_query = f'''
        CREATE TABLE IF NOT EXISTS manzana1 (
            id INTEGER PRIMARY KEY,
            date TEXT,
            prev_close TEXT,
            {columns}
        )
    '''
    print(create_table_query)
    #ok now actually create the database
    conn = sqlite3.connect('manzana1.db')
    cursor = conn.cursor()
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()