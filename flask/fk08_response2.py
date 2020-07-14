from flask import Flask, Response, make_response

app=Flask(__name__)

@app.route("/")
def response_test():
    custom_response = Response("★Custom_Response",200,
    {"Program":"Flask Web Framework"}
    )
    return make_response(custom_response)

@app.before_first_request
def before_first_request():
    print("0.앱이 기동되고 나서 첫번째 http 요청에만 응답합니다.")

@app.after_request
def after_reques(response):
    print("1.매 HTTP 요청이 처리되기 나서 실행됩니다.")
    return response

@app.before_request
def before_request():
    print("2.매 HTTP 요청이 처리되기 전에 실행됩니다.")

@app.teardown_request
def teardown_request(exception):
    print("3.매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다.")


@app.teardown_appcontext
def teardown_appcontext(exception):
    print("4.매 HTTP 요청의 애플리케이션 콘텍스트가 종료될 때 실행된다.")


if __name__=="__main__":
    app.run(debug=True)


