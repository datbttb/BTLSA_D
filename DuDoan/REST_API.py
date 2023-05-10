# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:54:55 2022

@author: Asus
"""

import pickle
from flask import Flask, request, json, jsonify
import numpy as np
import pandas as pd
app = Flask(__name__)
#---the filename of the saved model---
filename = 'diabetes.sav'
#---load the saved model---
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/diabetes/v1/predict', methods=['POST'])
def predict():
# =============================================================================
#     d = {'weight':[70], 'age': [23], 'height': [175]}
#     Me = pd.DataFrame(data=d)
# =============================================================================
#    features = request.json
    d = {'weight':[request.json.get('cannang')], 'age': [request.json.get('tuoi')], 'height': [request.json.get('chieucao')]}
    Me = pd.DataFrame(data=d)
    prediction = loaded_model.predict(Me)
    response = {}
    response['prediction'] = prediction[0]
    return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)