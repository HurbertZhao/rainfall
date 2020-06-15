from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np


df = pd.read_csv('station178_filled.csv')
X = df[['prcp','stp','smax', 'smin', 'temp','dewp','tmax','dmax','tmin','dmin','hmdy','hmax','hmin','wdsp','wdct','gust']]
Y = df["prcp"]
names = ['prcp','stp','smax', 'smin', 'temp','dewp','tmax','dmax','tmin','dmin','hmdy','hmax','hmin','wdsp','wdct','gust']

X = np.array(X.values)
data_y = np.array(Y.values)
label = np.zeros_like(data_y)
label[data_y > 0] = 1
label[data_y > 10] = 2
label[data_y > 25] = 3
print(X.shape)
clf = ExtraTreesClassifier()
clf = clf.fit(X, label)
index = np.argsort(clf.feature_importances_)
print(clf.feature_importances_)
print(index)

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)


