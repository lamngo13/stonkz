import sqlite3

stocklist = ["RTX", 'NOC', 'GD', 'LDOS', 'KBR', 'BWXT', 'RKLB', 'LMT']

s = 'RTX'
#ALLONE THIS IS A SINGLETON

conn = sqlite3.connect(s+'.db')
cursor = conn.cursor()

# Execute the DELETE statement for identical entries
query = f'''
        DELETE FROM {s}
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM {s}
            GROUP BY date
            HAVING COUNT(*) > 1
        )
    '''

cursor.execute(query)
# Commit the changes and close the connection
conn.commit()
conn.close()