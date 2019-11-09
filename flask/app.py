import h2o, sys, logging, urllib3, requests, json

from flask import Flask, render_template, url_for, request, json

from h2o.automl import H2OAutoML

h2o.init()

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
     return render_template('index.html')

@app.route("/test")
def test():
    return render_template('test.html')

@app.route("/predict", methods = ['GET', 'POST'])
def predict():

    data = request.json

    # load the model
    saved_model = h2o.load_model('static/js/model.dms')
    
    test_ex = [[data['inputGRE'], data['inputToefl'], data['inputUniRating'], data['inputSOP'], data['inputLOR'], data['inputCGPA'], data['inputResearch']]]
    column_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    test_frame = h2o.H2OFrame(test_ex, column_names = column_names)

    chance = saved_model.predict(test_frame)
    # x = request.json

    print(str(chance.flatten()))
    return str(chance.flatten())

if   __name__ == '__main__':
    app.run(debug=True)