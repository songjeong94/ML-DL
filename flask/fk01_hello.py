from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello333():
    return "<h1>hello youngsun world</h1>"

@app.route('/bit/bitcamp')
def hello334():
    return "<h1>hello bitcamp computer world </h1>"

@app.route('/gema')
def hello335():
    return "<h1>hello gema world </h1>"

if __name__ =='__main__':
    app.run(host='127.0.0.1', port=8888, debug=True)
