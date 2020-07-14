from flask import Flask
app=Flask(__name__)

from flask import make_response


@app.route("/<name>")
def index(name):
    response = make_response("<h1> Can you follow a step? </h1>")
    response.set_cookie("answer","42")
    return response

if __name__=="__main__":
    app.run(host="127.0.0.1",port=9898,debug=True)
    # print(hello332())