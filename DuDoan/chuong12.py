# Data Manipulation and Linear Algebra
import pandas as pd
import numpy as np

# Plots
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model, neighbors, ensemble, tree

df = pd.read_csv('final_test.csv')
print(df.head())

print(df.dtypes)

print(df.isna().sum())

bins = [150, 155,160, 165, 170, 175, 180, 185, 190, 195, 200]
ax = sns.histplot(data=df, x='height', bins=bins, color=sns.color_palette('Set2')[2], linewidth=2)
ax.set(title='Histogram', xlabel='Height (cm)', ylabel='Count')

ax = sns.histplot(data=df, x='age', color=sns.color_palette('Set2')[2], linewidth=2)
ax.set(title='Histogram', xlabel='Age', ylabel='Count')

plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data=df, x='height', y='weight',   hue='size', palette='Set2', s=60)
ax.set(xlabel='Height (cm)', ylabel='Weight (kg)')
ax.legend(title='Size', title_fontsize = 12) 
plt.show()

df = df.loc[(df['age'] >= 20.0) & (df['age'] <= 60)]
print('Min value: ', df['age'].min())
print('Max value: ',df['age'].max())

df['size'].value_counts()

print('Average height for XXS clothes: ',df.loc[df['size'] == 'XXL']['height'].mean(), ' and average weight: ', df.loc[df['size'] == 'XXS']['weight'].mean())
print('Average height for S clothes: ',df.loc[df['size'] == 'S']['height'].mean(),' and average weight: ', df.loc[df['size'] == 'S']['weight'].mean())
print('Average height for M clothes: ',df.loc[df['size'] == 'M']['height'].mean(),' and average weight: ', df.loc[df['size'] == 'M']['weight'].mean())
print('Average height for L clothes: ',df.loc[df['size'] == 'L']['height'].mean(),' and average weight: ', df.loc[df['size'] == 'L']['weight'].mean())
print('Average height for XL clothes: ',df.loc[df['size'] == 'XL']['height'].mean(),' and average weight: ', df.loc[df['size'] == 'XL']['weight'].mean())
print('Average height for XXL clothes: ',df.loc[df['size'] == 'XXL']['height'].mean(),' and average weight: ', df.loc[df['size'] == 'XXL']['weight'].mean())
print('Average height for XXXL clothes: ',df.loc[df['size'] == 'XXXL']['height'].mean(),' and average weight: ', df.loc[df['size'] == 'XXXL']['weight'].mean())

df = df[df['size']!= 'XXL']
df["size"].replace({"XXS": "XS", "XXXL": "XXL"}, inplace=True)
df["size"].value_counts()

plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data=df, x='height', y='weight',   hue='size', palette='Set2', s=60)
ax.set(xlabel='Height (cm)', ylabel='Weight (kg)')
ax.legend(title='Size', title_fontsize = 12) 
plt.show()

# =============================================================================
# Bắt đầu trainning sử dụng Decision Tree
# =============================================================================

from sklearn.model_selection import train_test_split

X = df.iloc[:,:-1]
y = df['size']

X=np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:")
print(accuracy_score(y_test, predictions))

import pickle
filename = 'diabetes.sav'


pickle.dump(model, open(filename, 'wb'))

#---load the model from disk---
loaded_model = pickle.load(open(filename, 'rb'))
d = {'weight':[70], 'age': [23], 'height': [175]}
Me = pd.DataFrame(data=d)
prediction = loaded_model.predict(Me)[0]
print(prediction)
    














