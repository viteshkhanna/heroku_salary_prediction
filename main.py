from logging import debug
from flask import Flask , render_template , request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('hiring._model.pkl')
@app.route('/hello')
def Welcome():
    return render_template('base.html')

@app.route('/predict', methods = ['POST'])
def predict():
    exp = request.form.get('experience')
    score = request.form.get('test_score')
    interview_score = request.form.get('interview_score')


    prediction  = model.predict([[int(exp) , int(score), int(interview_score)]])
    output = round(prediction[0],2)

    return render_template('base.html' , prediction_text = f"Employee salary will be ${output}")

app.run(debug=True)