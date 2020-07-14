from flask import Flask
app=Flask(__name__)

@app.route("/")
def hello333():
    return "<h1>hello world</h1>"


@app.route("/gema")
def hello332():
    return "<h1>hello gema world</h1>"

@app.route("/bit")
def hello335():
    return "<h1>hello bitcamp world</h1>"

@app.route("/bit/bitcamp")
def hello3334():
    return "<h1>hello bitcamp world</h1>"


if __name__=="__main__":
    app.run()
# * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
# 기본값