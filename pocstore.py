import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date

shouldwrite = True #set to False if we get bad data from api 

areweidentical = False

filesModified= ['bruh']


#get current date
today_date = datetime.now().date()
#datetime(year, month, day)
test_date = datetime(2023, 10, 12).date()
#^use this to hardcode specific dates. 


#now im trying to figure out how to backfill by thinking about how to hardcode a specific date.

#date like this 2023-08-23
#print(str(today_date))
#USE THIS TO BACKFILL MISSED DAYS 
#today_date = date(2023,8,25)

#print(str(today_date))

#this is a SMALL db to ensure that we don't overwrite identical days, prob unnecesary but hey who knows
zconn = sqlite3.connect("identical.db")
        # Create a cursor
zcursor = zconn.cursor()
# Create a table
zcreate_table_query = '''
        CREATE TABLE IF NOT EXISTS holder (
            id INTEGER PRIMARY KEY,
            date TEXT
        )
        '''
zcursor.execute(zcreate_table_query)
zconn.commit()


zcheckidenticalquery = """
SELECT EXISTS(SELECT 1 FROM holder WHERE date = ?)
"""
zcursor.execute(zcheckidenticalquery, (str(today_date),))
result = zcursor.fetchone()[0] #prob result true if exists

if not result:
    #write to db
    zdatewriter = ''' INSERT INTO holder (date) VALUES (?)'''
    zcursor.execute(zdatewriter, (str(today_date),))
    zconn.commit()
    filesModified.append("identicaldb")

zconn.close()

#TODO EXIT IF IDENTICAL

if (result):
    exit()




# Ticker symbol of the stock you want to fetch data for
#defense stock tickers
#ticker = 'AAPL'  # Example ticker symbol
stocklist = ["APPL", "RTX,", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']

# Define a date range around the desired date
start_date = today_date - timedelta(days=0)
end_date = today_date + timedelta(days=1)

#backfilling 

# Fetch data for the date range

#put this in a big huge big ol try except in the case that the api fails or something 
for s in stocklist:
    shouldwrite = True
    ticker = s
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        data = data.to_dict()

        cols = {'open': "Open", 'high': "High", 'low': "Low", 'close': "Close", 'volume': "Volume"}
        for c in cols:
            #get and format the vals from the incoming
            val = data[cols[c]]
            key = list(val.keys())[0]
            val = val[key]

            #reappend to dict 'cols'
            cols[c] = val

        #now theoretically each key has the val of the thing it describes 
        #I'm doing this like a barbarian but I will improve later
        #jackfruit
        name = s
        open = str(cols['open'])
        high = str(cols['high'])
        low = str(cols['low'])
        close = str(cols['close'])
        volume = str(cols['volume'])
    except:
        shouldwrite = False
        print("ticker: "+  ticker)


    #ONLY DO THIS IF THE DATA WE GOT IS GOOD 
    #WRITE TO THE ALL IN ONE DATABASE
    if shouldwrite:
        conn = sqlite3.connect('ALLONE.db')
        # Create a cursor
        cursor = conn.cursor()

        # Create a table
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS ALLONE (
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
        cursor.execute(create_table_query)

        # Insert data
        insert_data_query = '''
        INSERT INTO ALLONE (name, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        data = (name, today_date, open, high, low, close, volume)
        cursor.execute(insert_data_query, data)

        # Commit changes and close the connection
        conn.commit()
        conn.close()



        #NOW EACH ONE GETS ITS OWN DB TOO HEHE!!!!!!!!!!!!!!!!!!!!!!!
        #i think this is create if not exists
        conn = sqlite3.connect(str(name)+".db")
        # Create a cursor
        cursor = conn.cursor()

        # Create a table
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS ''' + str(name) + ''' (
            id INTEGER PRIMARY KEY,
            date TEXT,
            open TEXT,
            high TEXT,
            low TEXT,
            close TEXT,
            volume TEXT
        )
        '''
        cursor.execute(create_table_query)

        #check to see if this entry already exists 

        # Insert data
        insert_data_query = '''
        INSERT INTO ''' + str(name) + ''' (date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)
        '''
        data = (today_date, open, high, low, close, volume)
        cursor.execute(insert_data_query, data)

        # Commit changes and close the connection
        conn.commit()
        filesModified.append(name)
        conn.close()

print("FILES MODIFIED: ")
print(filesModified)
#for fm in filesModified:
#    print(filesModified.index(fm))





