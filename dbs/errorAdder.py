import sqlite3

# Connect to the database (if it doesn't exist, SQLite will create it)
conn = sqlite3.connect('wpred.db')

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Add a new column 'prediction' with the same data as 'close'
try:
    cursor.execute('''ALTER TABLE ALLONE ADD COLUMN prediction REAL;''')  # Replace your_table_name with the actual table name
    cursor.execute('''UPDATE ALLONE SET prediction = close;''')  # Replace your_table_name with the actual table name
    print("Column 'prediction' added successfully.")
except sqlite3.Error as e:
    print("Error:", e)

# Commit the changes and close the connection
conn.commit()
conn.close()
