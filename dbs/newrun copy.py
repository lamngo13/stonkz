import sqlite3
import yfinance as yf
from datetime import datetime, timedelta, date

#THIS GETS NEW STOCK DATA AND PUTS IT IN ALLONE.DB
print("BE KIND TO OTHERS ILY <3")

shouldwrite = True #set to False if we get bad data from api 

areweidentical = False

filesModified= ['start']


#get current date
today_date = datetime.now().date()

test_date = datetime(2023, 10, 30).date()
#YEAR MONTH DAY
#^use this to hardcode specific dates. 

today_date = test_date




# Ticker symbol of the stock you want to fetch data for
#defense stock tickers
#ticker = 'AAPL'  # Example ticker symbol
stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']
individualdbslister = stocklist


# Define a date range around the desired date
start_date = today_date - timedelta(days=0)
end_date = today_date + timedelta(days=1)
#end date is j one day later 

#trying a new way, j a query to make sure we don't get duplicates 
#put this in a big huge big ol try except in the case that the api fails or something 
for s in stocklist:
    shouldwrite = True
    ticker = s

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        #here is where the api magic happens

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

        #check to see if identical stock 
        identicalquery = '''SELECT * FROM ALLONE WHERE name = ? AND date = ?'''
        cursor.execute(identicalquery, (s, today_date))
        present = cursor.fetchone()
        #this should be None if that entry does NOT exist 

        if present is None:
            # Insert data
            insert_data_query = '''
            INSERT INTO ALLONE (name, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            data = (name, today_date, open, high, low, close, volume)
            cursor.execute(insert_data_query, data)
            #we successfully wrote, so we will mark this for ease of viewing
            filesModified.append(s)
        else:
            print("entry for " +str(s) + " already exists")

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        #NOW EACH ONE GETS ITS OWN DB TOO HEHE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        #check to see if identical stock 
        identicalquery = '''SELECT * FROM ''' + str(name) + ''' WHERE date = ?'''
        cursor.execute(identicalquery, (today_date,))
        present = cursor.fetchone()
        #this should be None if that entry does NOT exist 

        if present is None:
            # Insert data
            insert_data_query = '''
            INSERT INTO ''' + str(name) + ''' (date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)
            '''
            data = (today_date, open, high, low, close, volume)
            cursor.execute(insert_data_query, data)
            #we successfully wrote, so we will mark this for ease of viewing
            individualdbslister.append(s)
        else:
            print("entry for " + str(s) + " already exists -- individual db")

        # Commit changes and close the connection
        conn.commit()
        conn.close()


undone = [stock for stock in stocklist if stock not in filesModified]
#this notes all the stocks that are IN stocklist but NOT modified 

print("ALL ONE DB: ")
print("stocks noted: ")
print(filesModified)
print("stocks left out: ")
print(undone)
#now to individual dbs
print("INDIVIDUAL DBS")
print("stocks noted: ")
print(individualdbslister)
zundone = [stock for stock in stocklist if stock not in individualdbslister]
print("stocks left out: ")
print(zundone)






