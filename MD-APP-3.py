#Lets Create the WebApp for Hiring dataset -
import numpy as np
from flask import Flask,request,render_template,jsonify
import joblib

app3 = Flask(__name__)
pickle_file = joblib.load('train_md3.pkl')

@app3.route('/')
def home():
    return render_template('index1.html')

@app3.route('/predict',methods=['POST'])
def predict():
    #For Rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = pickle_file.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index1.html',prediction_text = 'Employee Salary should be INR {}'.format(output))

@app3.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = pickle_file.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
	app3.run(debug=True)

