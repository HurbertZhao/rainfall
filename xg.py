import xgboost as xgb
import torch
import numpy as np
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from GetData import getdata
from sklearn import metrics
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
    train = xgb.DMatrix(train_x, label=train_y)

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 4,                # 类别数，与 multisoftmax 并用
        'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 7,               # 构建树的深度，越大越容易过拟合
        'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,              # 随机采样训练样本
        'colsample_bytree': 0.7,       # 生成树时进行的列采样
        'min_child_weight': 3,
        'eta': 0.001,                  # 如同学习率
        'seed': 1000,
        'nthread': 4,                  # cpu 线程数
    }

    num_round = 7
    model = xgb.train(params, train, num_round)
    joblib.dump(model, path + ".joblib.dat")

def test(model,path):
    df = pd.read_csv('./testset/' + path + '.csv')
    chunk_size_x = 1
    chunk_size_y = 1

    test_x, test_y, scaler = \
        getdata_test(df, chunk_size_x, chunk_size_y)
    test_x = xgb.DMatrix(test_x)
    y_pred = model.predict(test_x)
    predictions = [round(value) for value in y_pred]

    kappa = cohen_kappa_score(test_y, predictions)
    print("kappa", kappa)
    accuracy = accuracy_score(test_y, predictions)
    print("accuracy", accuracy)
    recall = metrics.recall_score(test_y, predictions, average='macro')
    print('recall', recall)


if __name__ == "__main__":
    trainSets = ['station178_filled','station303_filled','station304_filled','station305_filled','station306_filled']
    testSets = ['station1','station2','station3','station4','station5']
    for i in range(5):
        print(trainSets[i]+' start training')
        train(trainSets[i])
        print('XGBOOST trained')

    # for i in range(5):
    #     print('\n' + trainSets[i] + ' start testing')
    #     model = joblib.load(trainSets[i] + ".joblib.dat")
    #     print('XGBOOST result:')
    #     test(model,testSets[i])
