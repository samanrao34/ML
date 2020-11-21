import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#create app and load the model
app = Flask(__name__)
model = pickle.load(open('scv_trained_model.pk1','rb'))

#Route to handle app
@app.route('/')
def home():
    return render_template('index.html')

 @app.route('/predict',methods=['POST'])
 def predict():
     inputs = [] #declaring input array

     inputs.append(request.form['pclass'])
     inputs.append(request.form['gender'])
     inputs.append(request.form['siblings'])
     inputs.append(request.form['embarked'])

     final_inputs = [np.array(inputs)]
     prediction = model.predict(final_inputs)

     if(prediction[0] == 1):
         return render_template('index.html',predicted_result = 'Survived')
    if(prediction[0] == 0):
         return render_template('index.html',predicted_result = 'Not Survived')

if __name__ == "__main__":
    app.run(debug=True)
    