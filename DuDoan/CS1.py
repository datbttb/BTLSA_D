# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 22:52:34 2022

@author: Admin
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 21:40:45 2022

@author: Asus
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
df = pd.read_csv('diabetes.csv')
df.info()

#---check for null values---
print("Nulls")
print("=====")
print(df.isnull().sum())

#---check for 0s---
print("0s")
print("==")
print(df.eq(0).sum())

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] =df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)

df.fillna(df.mean(numeric_only=True), inplace = True)

print(df.eq(0).sum())

corr = df.corr()

# =============================================================================
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(10, 10))
# cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(df.columns),1)
# ax.set_xticks(ticks)
# ax.set_xticklabels(df.columns)
# plt.xticks(rotation = 90)
# ax.set_yticks(ticks)
# ax.set_yticklabels(df.columns)
# 
# #---print the correlation factor---
# for i in range(df.shape[1]):
#     for j in range(9):
#         text = ax.text(j, i, round(corr.iloc[i][j],2),
#                        ha="center", va="center", color="w")
# plt.show()
# =============================================================================


#---print the top 4 correlation values---
print(df.corr().nlargest(4, 'Outcome').values[:,8])

#---features---
x = df[['Glucose','BMI','Age']]
#---label---
Y = df.iloc[:,8]
print(Y)
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.3, random_state=5)
# =============================================================================
# pl=2
# y_train=np_utils.to_categorical(Y_train,pl);
# y_test=np_utils.to_categorical(Y_test,pl);
# 
# =============================================================================


#Sử dụng deeplearning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(Dense(100, input_shape=(3,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer="adam",loss = "categorical_crossentropy",metrics = ["accuracy"])
history=model.fit(x_train, Y_train,epochs=100,batch_size=10,validation_data=(x_test,Y_test), verbose=0)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



