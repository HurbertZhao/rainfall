import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from pandas import DataFrame
from pandas import concat
from itertools import chain
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from data_operate import getDataSet
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection  import train_test_split

def getdata(df, chunk_size_x,chunk_size_y):
    list = df['Unnamed: 0'].tolist()
    data_set = df.iloc[:, 15:].values
    data_set = data_set.astype('float64')
    #归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()

    train_data_set = np.array(data_set)
    #得到数据集
    reframed_train_data_set = np.array(getDataSet(train_data_set, chunk_size_x, chunk_size_y,True,list).values)
    reframed_train_data_set = reframed_train_data_set[:,:16*chunk_size_x+1]
    # 数据集划分,选取前60%天的数据作为训练集,中间20%天作为验证集,其余的作为测试集
    train_days = int(len(reframed_train_data_set) * 0.6)
    valid_days = int(len(reframed_train_data_set) * 0.2)

    data = reframed_train_data_set[:,:-chunk_size_y]
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
    label = to_categorical(label, num_classes=4)
    train_y = label[:train_days]
    valid_y = label[train_days:train_days + valid_days]
    test_y = label[train_days + valid_days:]

    # 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征维度：16]
    train_x = train_x.reshape((train_x.shape[0], chunk_size_x, 16))
    valid_x = valid_x.reshape((valid_x.shape[0], chunk_size_x, 16))
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 16))

    return train_x, train_y, valid_x, valid_y, test_x, test_y , scaler

def gettrain_rand(df, chunk_size_x,chunk_size_y):
    list = df['Unnamed: 0'].tolist()
    trainsize = int(df.shape[0]*0.9)
    data_set = df.iloc[:trainsize, 15:].values
    data_set = data_set.astype('float64')
    # 归一化
    scaler = StandardScaler()
    train_data_set = np.array(data_set)
    # 得到数据集
    reframed_train_data_set = np.array(getDataSet(train_data_set, chunk_size_x, chunk_size_y, True, list).values)
    reframed_train_data_set = reframed_train_data_set[:, :16*chunk_size_x + 1]
    data = reframed_train_data_set[:, :-chunk_size_y]
    scaler.fit(data)
    data = scaler.transform(data)
    data_y = reframed_train_data_set[:, -chunk_size_y:]
    label = np.zeros_like(data_y)
    label[data_y > 0] = 1
    label[data_y > 0.5] = 2
    label[data_y > 1.5] = 3
    label = to_categorical(label, num_classes=4)
    train_x,rest_x,train_y,rest_y = train_test_split(data,label,test_size = 0.2,random_state= 0)
    valid_x,test_x,valid_y,test_y = train_test_split(rest_x,rest_y,test_size = 0.5,random_state= 0)

    train_x = train_x.reshape((train_x.shape[0], chunk_size_x, 16))
    valid_x = valid_x.reshape((valid_x.shape[0], chunk_size_x, 16))
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 16))

    return train_x,train_y,valid_x,valid_y,test_x,test_y

def lstm_model(train_X,chunk_size_y):
    model = Sequential()
    #第一层
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    # 第二层
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    # 第三层 因为是回归问题所以使用linear
    model.add(Dense(4 * chunk_size_y, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

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
    # label = to_categorical(label, num_classes=4)
    test_y = label
    # 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征维度：16]
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 16))
    return test_x, test_y , scaler

def train_vote(path):
    df = pd.read_csv(path + '.csv')
    chunk_size_x = 6
    chunk_size_y = 1
    input_epochs = 20
    input_batch_size = 100

    train_x_, label_y_, valid_x, valid_y, test_x, test_y = \
        gettrain_rand(df, chunk_size_x, chunk_size_y)
    for i in range(10):
        train_x, _,label_y,_ =  train_test_split(train_x_, label_y_,test_size = 0.4,random_state= None)
        model = lstm_model(train_x, chunk_size_y)
        model.fit(train_x, label_y, epochs=input_epochs, batch_size=input_batch_size,
                  validation_data=(valid_x, valid_y), verbose=2, shuffle=False)
        model.save(path + 'lstm' + str(i) + '.h5')
        # test_predict = model.predict(test_x)
        # pred_cls = np.argmax(test_predict, axis=1)
        # test_y_cl = np.argmax(test_y, axis=1)
        # kappa = cohen_kappa_score(test_y_cl, pred_cls)
        # print(kappa)
        # acc = accuracy_score(test_y_cl, pred_cls)
        # acc0 = accuracy_score(test_y_cl, np.zeros_like(test_y_cl))
        # print(acc, acc0)
        # recall = metrics.recall_score(test_y_cl, pred_cls, average='macro')
        # print('recall:%f' % recall)

def vote(path,pathtest):
    models = []
    for i in range(10):
        model = load_model(path + 'lstm' + str(i) + '.h5')
        models.append(model)

    df = pd.read_csv('./testset/' + pathtest + '.csv')
    chunk_size_x = 6
    chunk_size_y = 1
    test_x, test_y,scaler =  getdata_test(df, chunk_size_x, chunk_size_y)

    pred_cls = np.zeros((test_x.shape[0],10))
    for i in range(10):
        model = models[i]
        test_predict = model.predict(test_x)
        pred_cls[:,i] = np.argmax(test_predict, axis=1)
    pred_cl = np.zeros((test_x.shape[0],1))
    for i in range(test_x.shape[0]):
        counts = np.bincount(pred_cls[i,:].astype(np.int32))
        pred_cl[i] = np.argmax(counts)

    kappa = cohen_kappa_score(test_y, pred_cl)
    print('kappa:%f'%kappa)
    acc = accuracy_score(test_y, pred_cl)
    print('accuracy:%f'%acc)
    recall = metrics.recall_score(test_y, pred_cl, average='macro')
    print('recall:%f' % recall)


if __name__ == '__main__':
    trainSets = ['station178_filled','station303_filled','station304_filled','station305_filled','station306_filled']
    testSets = ['station1','station2','station3','station4','station5']
    for i in range(5):
        print(trainSets[i]+' start training')
        train_vote(trainSets[i])
        print('Bagging trained')

    # for i in range(5):
    #     print('\n' + trainSets[i]+' start testing')
    #     vote(trainSets[i],testSets[i])
    #     print('Bagging result:')

