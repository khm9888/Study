from flask import Flask
app=Flask(__name__)

@app.route("/<name>")
def user(name):
    return f"<h1>hello, {name} </h1>"
    
@app.route("/user/<name>")
def user2(name):
    return f"<h1>hello, user/{name} </h1>"


if __name__=="__main__":
    app.run(host="127.0.0.1",port=8888,debug=True)
    # print(hello332())