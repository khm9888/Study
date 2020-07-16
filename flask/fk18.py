from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general")
print("fetchall: ",cursor.fetchall())

@app.route("/")
def run():
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general")
    rows = c.fetchall()
    return render_template("board_index.html", rows = rows)

@app.route("/modi")
def modi():
    id = request.args.get("id")
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general where id = "+str(id))
    rows = c.fetchall();
    return render_template("board_modi.html", rows = rows)

@app.route("/addred", methods = ["POST", "GET"])
def addred():
    if request.method == 'POST':
        print(request.form['war'])
        print(request.form['id'])
        try:
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect("./data/wanggun.db") as con:
                print("con")
                cur = con.cursor()
                cur.execute("UPDATE general SET war="+str(war)+" WHERE id="+str(id))
                print("con2")
                con.commit()
                msg="정상적으로 입력되었습니다"

        except:
            con.rollback()
            msg="입력과정에서 에러가 발생했습니다."

        finally:
            return render_template("board_result.html",msg = msg)
            con.close()

app.run(host="127.0.0.1",port=5000,debug=False)