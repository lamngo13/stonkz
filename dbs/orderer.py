import sqlite3

# Connect to the database
conn = sqlite3.connect('ALLONE.db')
cursor = conn.cursor()

# Query to reorder the table by date in ascending order and update the original table
query = '''
    CREATE TABLE temp_table AS
    SELECT * FROM ALLONE
    ORDER BY DATE(date) ASC;
    
    DROP TABLE ALLONE;
    
    ALTER TABLE temp_table RENAME TO ALLONE;
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
