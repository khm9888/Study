import sqlite3

conn = sqlite3.connect("test.db") # 만든 적 없는  test.db 라는 데이터가 자동으로 생성된다

cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT, 
                  FoodName TEXT, Company TEXT, Price INTEGER)""") # supermarket이라는 테이블이 없다면 생성하라

# column : ItemNo, Category, FoodName, Company, Price

sql = "DELETE FROM supermarket" # supermaker의 내용을 지워버린다
cursor.execute(sql)

# 데이터 삽입 (현재 컬럼명, 테이블 모양만 갖춰진 상태)
sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (1, '과일', '자몽', '마트', 1500))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (2, '음료', '망고주스', '편의점', 1000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (33, '육류', '소고기', '하나로마트', 10000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (4, '박카스', '약', '약국', 500))


# 데이터 조회
sql = "SELECT * FROM supermarket"
cursor.execute(sql)

rows = cursor.fetchall()


for row in rows :
    print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
          str(row[3]) + " " + str(row[4]))
