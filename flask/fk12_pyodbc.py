import pyodbc as pyo

server ="127.0.0.1"
database ="bitdb"
username="bit2"
password="1234"


conn = pyo.connect(f"""DRIVER={{ODBC Driver 17 for SQL Server}}; SERVER={server}; PORT=1433;
 DATEBASE={database};
 UID={username}; 
 PWD={password};""")

cursor = conn.cursor()

tsql = 'select* from iris'

cursor.execute(tsql)

# for _ in range(150):
cnt=0
while True:
    
    row =cursor.fetchone()#한줄을 가져온다.
    print(f"cnt:{cnt}, row:{row}")
    if not row:
        break
    cnt+=1

conn.close()