import pymssql as ms

conn = ms.connect(server="127.0.0.1",user="bit2",password="1234")

cursor = conn.cursor()

cursor.execute("select * from iris")

# for _ in range(150):
cnt=0
while True:
    
    row =cursor.fetchone()#한줄을 가져온다.
    print(f"cnt:{cnt}, row:{row}")
    if not row:
        break
    cnt+=1

conn.close()