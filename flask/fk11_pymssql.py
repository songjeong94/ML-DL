import pymssql as ms
print('잘 접속됨')
conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',
            database='bitdb',port= 49680)

cursor = conn.cursor()

# cursor.execute("SELECT * FROM iris;")
# cursor.execute("SELECT * FROM wine;")
cursor.execute("SELECT * FROM sonar;")
row = cursor.fetchone()

print("column1: %s, column2: %s" %(row[0], row[1]))
row = cursor.fetchone() 

conn.close()

