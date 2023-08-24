#yuh yeet
import mysql.connector

# Connect to MySQL Server
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="SaraiAlvarengaNgo13*"
)

# Create a new database
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE mydatabase")

# Connect to the new database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="mydatabase"
)

# Create a new table
mycursor = mydb.cursor()
mycursor.execute("CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255))")

# Insert data into the table
sql = "INSERT INTO customers (name, email) VALUES (%s, %s)"
val = ("John Doe", "john@example.com")
mycursor.execute(sql, val)
mydb.commit()

# Query the data
mycursor.execute("SELECT * FROM customers")
result = mycursor.fetchall()
for row in result:
    print(row)

# Close the connection
mydb.close()
