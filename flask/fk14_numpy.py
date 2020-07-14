import pymssql as ms
import numpy as np

conn = ms.connect(server="127.0.0.1",user="bit2",password="1234")

cursor = conn.cursor()

cursor.execute("select * from iris")

# for _ in range(150):
rows = np.asarray(cursor.fetchall())
print(rows)

conn.close()

np.save('data/tesk_flask_iris2.npy',rows)