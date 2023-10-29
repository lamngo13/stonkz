import sqlite3

# Connect to the database
conn = sqlite3.connect('wpred.db')
cursor = conn.cursor()

# Query to order the table by date in ascending order
query = '''
    SELECT * FROM ALLONE
    ORDER BY date ASC;
'''

# Execute the query
cursor.execute(query)
conn.commit()

# Fetch the results if needed
results = cursor.fetchall()

# Process the results as needed
for row in results:
    print(row)

# Close the connection
conn.close()
