# -*- coding: utf-8 -*-
"""
Created on Thu May 11 01:25:39 2023

@author: Admin
"""

import pandas as pd
import pickle
filename = 'diabetes.sav'
#---load the model from disk---
loaded_model = pickle.load(open(filename, 'rb'))
d = {'weight':[70], 'age': [23], 'height': [175]}
Me = pd.DataFrame(data=d)
prediction = loaded_model.predict(Me)[0]
print(prediction)