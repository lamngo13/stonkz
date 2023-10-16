import sqlite3

#ALLONE THIS IS A SINGLETON
conn = sqlite3.connect('ALLONE.db')
cursor = conn.cursor()

# Execute the DELETE statement for identical entries
cursor.execute("""
    DELETE FROM bruh
    WHERE rowid NOT IN (
        SELECT MIN(rowid)
        FROM bruh
        GROUP BY name, date
        HAVING COUNT(*) > 1
    )
""")

# Commit the changes and close the connection
conn.commit()
conn.close()