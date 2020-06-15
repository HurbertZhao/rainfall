import torch
import numpy as np
import time
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from GetData import getdata
from data_operate import getDataSet
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def getdata_test(df, chunk_size_x,chunk_size_y):
    list = df['Unnamed: 0'].tolist()
    data_set = df.iloc[:, 15:].values
    data_set = data_set.astype('float64')
    #归一化
    scaler = StandardScaler()

    train_data_set = np.array(data_set)
    #得到数据集
    reframed_train_data_set = np.array(getDataSet(train_data_set, chunk_size_x, chunk_size_y,True,list).values)
    reframed_train_data_set = reframed_train_data_set[:,:16*chunk_size_x+1]

    data = reframed_train_data_set[:,:-chunk_size_y]
    scaler.fit(data)
    data = scaler.transform(data)
    test_x = data
    data_y = reframed_train_data_set[:, -chunk_size_y:]
    label = np.zeros_like(data_y)
    label[data_y > 0] = 1
    label[data_y > 0.5] = 2
    label[data_y > 1.5] = 3
    test_y = label
    test_x = test_x.reshape((test_x.shape[0], 16))
    return test_x, test_y , scaler

def train(path):
    df = pd.read_csv(path + '.csv')
    chunk_size_x = 1
    chunk_size_y = 1
    train_x, train_y, valid_x, valid_y, test_x, test_y, scaler = getdata(df, chunk_size_x, chunk_size_y)

    clf = RandomForestClassifier(n_estimators=10, max_depth=9, criterion='gini')
    clf = clf.fit(train_x, train_y)
    joblib.dump(clf, path + 'RF.pkl')

def test(clf,path):
    df = pd.read_csv('./testset/' + path + '.csv')
    chunk_size_x = 1
    chunk_size_y = 1

    test_x, test_y, scaler = \
        getdata_test(df, chunk_size_x, chunk_size_y)

    y_pred = clf.predict(test_x)
    kappa = cohen_kappa_score(test_y, y_pred)
    print("kappa", kappa)
    accuracy = accuracy_score(test_y, y_pred)
    print("accuracy", accuracy)
    recall = metrics.recall_score(test_y, y_pred, average='macro')
    print('recall', recall)


if __name__ == "__main__":
    trainSets = ['station178_filled','station303_filled','station304_filled','station305_filled','station306_filled']
    testSets = ['station1','station2','station3','station4','station5']
    for i in range(5):
        print(trainSets[i]+' start training')
        train(trainSets[i])
        print('RandomForest trained')
    #
    # for i in range(5):
    #     print('\n' + trainSets[i]+' start testing')
    #     model = joblib.load(trainSets[i] + 'RF.pkl')
    #     print('RandomForest result:')
    #     test(model,testSets[i])