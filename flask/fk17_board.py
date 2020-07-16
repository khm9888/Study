from flask import Flask, render_template, request
import sqlite3
import numpy as np

app = Flask(__name__)

conn = sqlite3.connect("./data/wanggun.db")
cursor1 = conn.cursor()
cursor1.execute("SELECT * from general")
print(np.asarray(cursor1.fetchall()))

@app.route("/")
def run():
    conn = sqlite3.connect("./data/wanggun.db")
    c = conn.cursor()
    c.execute("SELECT * FROM general")
    rows=c.fetchall()
    return render_template("board_index.html",rows=rows)


# 2. wanggun.db에서 모든 행을 땡겨오는 라우트 발동
@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    # WHERE~~ == ~~를 찾아서 실행해라
    c.execute(f"SELECT * FROM general WHERE id={id}")
    rows = c.fetchall();
    return render_template("board_modi.html", rows=rows)



@app.route('/addrec',methods=['POST',"GET"])
def addrec():
    if request.method =='POST':
        try:
            war = request.form["war"]
            id = request.form["id"]
            with sqlite3.connect("./data/wanggun.db") as conn:
                cur = conn.cursor()
                cur.execute(f"UPDATE general SET war={war} where id={id}")
                conn.commit()
                msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()
            msg = '입력과정에서 에러가 발생했습니다.'
        finally:
            return render_template("board_result.html",msg=msg)
            conn.close()
                      
if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5001,debug=False)
    

