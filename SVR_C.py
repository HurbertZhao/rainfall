import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from data_operate import getDataSet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.externals import joblib
from sklearn.metrics import cohen_kappa_score

def getdata(df, chunk_size_x,chunk_size_y):
    list = df['Unnamed: 0'].tolist()
    data_set = df.iloc[:, 15:].values
    data_set = data_set.astype('float64')

    train_data_set = np.array(data_set)
    #得到数据集
    reframed_train_data_set = np.array(getDataSet(train_data_set, chunk_size_x, chunk_size_y,True,list).values)
    reframed_train_data_set = reframed_train_data_set[:,:16*chunk_size_x+1]
    # 数据集划分,选取前70%天的数据作为训练集,中间20%天作为验证集,其余的作为测试集
    train_days = int(len(reframed_train_data_set) * 0.7)
    valid_days = int(len(reframed_train_data_set) * 0.2)

    # 归一化
    scaler = StandardScaler()
    data = reframed_train_data_set[:, :-chunk_size_y]
    scaler.fit(data)
    data = scaler.transform(data)
    train_x = data[:train_days, :]
    valid_x = data[train_days:train_days + valid_days, :]
    test_x = data[train_days + valid_days:, :]

    data_y = reframed_train_data_set[:, -chunk_size_y:]
    label = np.zeros_like(data_y)
    label[data_y > 0] = 1
    label[data_y > 0.5] = 2
    label[data_y > 1.5] = 3
    # label = to_categorical(label, num_classes=4)
    train_y = label[:train_days]
    valid_y = label[train_days:train_days + valid_days]
    test_y = label[train_days + valid_days:]

    train_x = train_x.reshape((train_x.shape[0], 16))
    valid_x = valid_x.reshape((valid_x.shape[0], 16))
    test_x = test_x.reshape((test_x.shape[0], 16))
    return train_x, train_y, valid_x, valid_y, test_x, test_y , scaler

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
    df = df[:30000]
    chunk_size_x = 1
    chunk_size_y = 1

    train_x, label_y, valid_x, valid_y, test_x, test_y, scaler = \
        getdata(df, chunk_size_x, chunk_size_y)

    svr = SVC(kernel = 'rbf', gamma='scale',C = 10)
    svr.fit(train_x, label_y)
    joblib.dump(svr, path + 'SVM.pkl')
    print("训练集得分:{}".format(svr.score(train_x, label_y)))
    print("验证集得分:{}".format(svr.score(valid_x, valid_y)))

    # print("测试集得分:{}".format(svr.score(test_x, test_y)))
    # test_predict = svr.predict(valid_x)
    # kappa = cohen_kappa_score(valid_y, test_predict)
    # print(kappa)
    # acc = accuracy_score(valid_y, test_predict)
    # acc0 = accuracy_score(valid_y, np.zeros_like(valid_y))
    # print(acc, acc0)
    # recall = metrics.recall_score(valid_y, test_predict, average='macro')
    # print('recall:%f' % recall)

def test(svr,path):
    df = pd.read_csv('./testset/' + path + '.csv')
    chunk_size_x = 1
    chunk_size_y = 1

    test_x, test_y, scaler = \
        getdata_test(df, chunk_size_x, chunk_size_y)

    test_predict = svr.predict(test_x)
    kappa = cohen_kappa_score(test_y, test_predict)
    print('kappa:%f'%kappa)
    acc = accuracy_score(test_y, test_predict)
    print('accuracy:%f'%acc)
    recall = metrics.recall_score(test_y, test_predict, average='macro')
    print('recall:%f' % recall)

if __name__ == '__main__':
    trainSets = ['station178_filled','station303_filled','station304_filled','station305_filled','station306_filled']
    testSets = ['station1','station2','station3','station4','station5']
    for i in range(5):
        print(trainSets[i]+' start training')
        train(trainSets[i])
        print('SVM trained')

    # for i in range(5):
    #     print(trainSets[i]+' start testing')
    #     model = joblib.load(trainSets[i] + 'SVM.pkl')
    #     print('SVM result:')
    #     test(model,testSets[i])


