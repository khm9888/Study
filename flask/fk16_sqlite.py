import sqlite3

conn = sqlite3.connect("test.db")

cursor = conn.cursor()

sql = "DELETE FROM supermarket" # supermaker의 내용을 지워버린다
cursor.execute(sql)

cursor.execute("""CREATE table if not exists supermarket(Itemno INTEGER,Category TEXT,
            FoodName TEXT, Company TEXT, Price INTEGER)""")


sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (1, '과일', '자몽', '마트', 1500))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (2, '음료', '망고주스', '편의점', 1000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (33, '육류', '소고기', '하나로마트', 10000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (4, '박카스', '약', '약국', 500))




sql = "SELECT * from supermarket"
cursor.execute(sql)

rows = cursor.fetchall()
# print(rows)
for i, row in enumerate(rows) :
    print(i, row)

conn.commit()
conn.close()
