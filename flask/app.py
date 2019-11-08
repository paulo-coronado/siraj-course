from flask import Flask, render_template, url_for, request, json

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
     return render_template('index.html')

@app.route("/test")
def test():
    return render_template('test.html')

if   __name__ == '__main__':
    app.run(debug=True)