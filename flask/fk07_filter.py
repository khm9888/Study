from flask import Flask,render_template

app=Flask(__name__)

@app.route("/user/<name>")
def user(name):
    return render_template("user2.html",name=name)

if __name__=="__main__":
    app.run(host="127.0.0.1",port=5001,debug=False)


