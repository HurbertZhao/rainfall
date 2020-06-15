import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
from data_operate import getDataSet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
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
    train_y = label[:train_days]
    valid_y = label[train_days:train_days + valid_days]
    test_y = label[train_days + valid_days:]

    train_x = train_x.reshape((train_x.shape[0], 16))
    valid_x = valid_x.reshape((valid_x.shape[0], 16))
    test_x = test_x.reshape((test_x.shape[0], 16))
    return train_x, train_y, valid_x, valid_y, test_x, test_y , scaler

def train(path):

    df = pd.read_csv(path + '.csv')
    chunk_size_x = 1
    chunk_size_y = 1

    train_x, label_y, valid_x, valid_y, test_x, test_y, scaler = \
        getdata(df, chunk_size_x, chunk_size_y)

    #base_estimator  = SVC(kernel = 'linear', gamma='scale',C = 10)
    base_estimator  = DecisionTreeClassifier(max_depth=7)
    n_estimators = [10,20]

    for n_estimator in n_estimators:
        adaB = AdaBoostClassifier(base_estimator=base_estimator,algorithm='SAMME', n_estimators=n_estimator, learning_rate=0.3,
                                  random_state=0)
        adaB.fit(train_x, label_y)
        joblib.dump(adaB,path + str(n_estimator) + '.joblib')


    #     y_train_pred = adaB.predict(train_x)
    #     acc_train = metrics.accuracy_score(label_y, y_train_pred)
    #     accs_train.append(acc_train)
    #
    #     y_test_pred = adaB.predict(test_x)
    #     acc_test = metrics.accuracy_score(test_y, y_test_pred)
    #     accs_test.append(acc_test)
    #     kappa = cohen_kappa_score(test_y, y_test_pred)
    #     kappas.append(kappa)
    #     recall = metrics.recall_score(test_y, y_test_pred, average='macro')
    #     recalls.append(recall)
    #
    # print(accs_train)
    # print(accs_test)
    # print(kappas)
    # print(recalls)
    #
    #
    # base_estimator  = DecisionTreeClassifier(max_depth=7)
    # base_estimator.fit(train_x, label_y)
    # y_test_pred = base_estimator.predict(test_x)
    # acc_test = metrics.accuracy_score(test_y, y_test_pred)
    # kappa = cohen_kappa_score(test_y, y_test_pred)
    # recall = metrics.recall_score(test_y, y_test_pred, average='macro')
    # print(acc_test, recall, kappa)

def test(adaB,path):

    df = pd.read_csv('./testset/' + path + '.csv')
    chunk_size_x = 1
    chunk_size_y = 1

    test_x, test_y, scaler = \
        getdata_test(df, chunk_size_x, chunk_size_y)
    y_test_pred = adaB.predict(test_x)



    acc_test = metrics.accuracy_score(test_y, y_test_pred)

    kappa = cohen_kappa_score(test_y, y_test_pred)

    recall = metrics.recall_score(test_y, y_test_pred, average='macro')

    print('Kappa:')
    print(kappa)
    print('Accuracy:')
    print(acc_test)
    print('Recall:')
    print(recall)


if __name__ == '__main__':
    trainSets = ['station178_filled', 'station303_filled', 'station304_filled', 'station305_filled',
                 'station306_filled']
    testSets = ['station1', 'station2', 'station3', 'station4', 'station5']
    for i in range(5):
        print(trainSets[i]+' start training')
        train(trainSets[i])
        print('ADABOOST trained')



