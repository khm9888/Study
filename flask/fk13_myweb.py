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

from flask import Flask, render_template

app=Flask(__name__)

@app.route("/sqltable")
def showsql():
    cursor.execute(tsql)
    return render_template("myweb.html",rows=cursor.fetchall())


if __name__ == "__main__":
    app.run(host=server,debug=True)

conn.close()