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
from keras.models import load_model
from keras.utils import to_categorical
from data_operate import getDataSet
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def getdata_1(df, chunk_size_x,chunk_size_y):
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
    # 数据集划分,选取前70%天的数据作为训练集,中间20%天作为验证集,其余的作为测试集
    train_days = int(len(reframed_train_data_set) * 0.7)
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
    # label[data_y > 0.5] = 2
    # label[data_y > 1.5] = 3
    label = to_categorical(label, num_classes=2)
    train_y = label[:train_days]
    valid_y = label[train_days:train_days + valid_days]
    test_y = label[train_days + valid_days:]

    # 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征维度：16]
    train_x = train_x.reshape((train_x.shape[0], chunk_size_x, 16))
    valid_x = valid_x.reshape((valid_x.shape[0], chunk_size_x, 16))
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 16))

    return train_x, train_y, valid_x, valid_y, test_x, test_y , scaler

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
    label[data_y > 5] = 2
    label[data_y > 10] = 3
    label = to_categorical(label, num_classes=4)
    train_y = label[:train_days]
    valid_y = label[train_days:train_days + valid_days]
    test_y = label[train_days + valid_days:]

    # 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征维度：16]
    train_x = train_x.reshape((train_x.shape[0], chunk_size_x, 16))
    valid_x = valid_x.reshape((valid_x.shape[0], chunk_size_x, 16))
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 16))

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
    # label = to_categorical(label, num_classes=4)
    test_y = label.reshape(-1)
    # 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征维度：16]
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 16))
    return test_x, test_y , scaler

def getdata_2(df, chunk_size_x,chunk_size_y):
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


    data_y = reframed_train_data_set[:, -chunk_size_y:]
    label = np.zeros_like(data_y)
    idx = data_y > 0
    # label[data_y > 0] = 1
    label[data_y > 0.5] = 1
    label[data_y > 1.5] = 2
    label = label[idx]
    train_days = int(len(label) * 0.6)
    valid_days = int(len(label) * 0.2)
    label = to_categorical(label, num_classes=3)
    train_y = label[:train_days]
    valid_y = label[train_days:train_days + valid_days]
    test_y = label[train_days + valid_days:]


    data = reframed_train_data_set[:,:-chunk_size_y]
    scaler.fit(data)
    data = scaler.transform(data)
    idx = idx.reshape(-1)
    data = data[idx]
    train_x = data[:train_days, :]
    valid_x = data[train_days:train_days + valid_days, :]
    test_x = data[train_days + valid_days:, :]



    # 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征维度：16]
    train_x = train_x.reshape((train_x.shape[0], chunk_size_x, 16))
    valid_x = valid_x.reshape((valid_x.shape[0], chunk_size_x, 16))
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 16))

    return train_x, train_y, valid_x, valid_y, test_x, test_y , scaler

def getdata_(df, chunk_size_x,chunk_size_y):
    list = df['Unnamed: 0'].tolist()
    data_set = df.iloc[:, 15].values
    data_set = data_set.astype('float64')
    #归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()

    train_data_set = np.array(data_set).reshape(-1,1)
    #得到数据集
    reframed_train_data_set = np.array(getDataSet(train_data_set, chunk_size_x, chunk_size_y,True,list).values)
    reframed_train_data_set = reframed_train_data_set[:,:chunk_size_x+1]
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
    train_x = train_x.reshape((train_x.shape[0], chunk_size_x, 1))
    valid_x = valid_x.reshape((valid_x.shape[0], chunk_size_x, 1))
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 1))

    return train_x, train_y, valid_x, valid_y, test_x, test_y , scaler

def lstm_model(train_X,chunk_size_y):
    model = Sequential()
    #第一层
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2])))
    # 第二层
    model.add(LSTM(64,activation='tanh', return_sequences=False))
    model.add(Dropout(0.5))
    # 第三层
    model.add(Dense(4 * chunk_size_y, activation='softmax'))
    Adam = optimizers.Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam)
    return model

def lstm_model_1(train_X,chunk_size_y):
    model = Sequential()
    #第一层
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2])))
    # 第二层
    model.add(LSTM(64,activation='tanh', return_sequences=False))
    model.add(Dropout(0.5))
    # 第三层
    model.add(Dense(2 * chunk_size_y, activation='sigmoid'))
    Adam = optimizers.Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam)
    return model

def lstm_model_2(train_X,chunk_size_y):
    model = Sequential()
    #第一层
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2])))
    # 第二层
    model.add(LSTM(64,activation='tanh', return_sequences=False))
    model.add(Dropout(0.5))
    # 第三层
    model.add(Dense(3 * chunk_size_y, activation='softmax'))
    Adam = optimizers.Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam)
    return model

def train(path):
    df = pd.read_csv(path+'.csv')
    chunk_size_x = 6
    chunk_size_y = 1
    input_epochs = 10
    input_batch_size = 50

    train_x, label_y, valid_x, valid_y, test_x, test_y, scaler= \
        getdata(df,chunk_size_x,chunk_size_y)

    model = lstm_model(train_x,chunk_size_y)
    elstop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(path+'.h5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=5, mode='auto')
    model.fit(train_x, label_y, epochs=input_epochs, batch_size=input_batch_size,
                    validation_data=(valid_x, valid_y), verbose=2, shuffle=False, callbacks=[elstop,checkpointer,reduce_lr])

    # model.save('lstm1_6_304_best.h5')

    test_predict = model.predict(test_x)
    pred_cls = np.argmax(test_predict,axis=1)
    test_y = np.argmax(test_y,axis=1)

    # kappa = cohen_kappa_score(test_y, pred_cls)
    # print(kappa)
    # acc = accuracy_score(test_y, pred_cls)
    # acc0 = accuracy_score(test_y, np.zeros_like(test_y))
    # print(acc, acc0)
    # recall = metrics.recall_score(test_y, pred_cls, average='macro')
    # print('recall:%f' % recall)

def train_(path):
    df = pd.read_csv(path+'.csv')
    chunk_size_x = 6
    chunk_size_y = 1
    input_epochs = 20
    input_batch_size = 100

    train_x, label_y, valid_x, valid_y, test_x, test_y, scaler= \
        getdata_(df,chunk_size_x,chunk_size_y)

    model = lstm_model(train_x,chunk_size_y)
    elstop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(path+'_.h5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=5, mode='auto')
    model.fit(train_x, label_y, epochs=input_epochs, batch_size=input_batch_size,
                    validation_data=(valid_x, valid_y), verbose=2, shuffle=False, callbacks=[elstop,checkpointer,reduce_lr])

    test_predict = model.predict(test_x)
    pred_cls = np.argmax(test_predict,axis=1)
    test_y = np.argmax(test_y,axis=1)

    kappa = cohen_kappa_score(test_y, pred_cls)
    print(kappa)
    acc = accuracy_score(test_y, pred_cls)
    acc0 = accuracy_score(test_y, np.zeros_like(test_y))
    print(acc, acc0)
    recall = metrics.recall_score(test_y, pred_cls, average='macro')
    print('recall:%f' % recall)

def train_1(path):
    df = pd.read_csv(path + '.csv')
    chunk_size_x = 6
    chunk_size_y = 1
    input_epochs = 100
    input_batch_size = 100

    train_x, label_y, valid_x, valid_y, test_x, test_y, scaler= \
        getdata_1(df,chunk_size_x,chunk_size_y)

    model = lstm_model_1(train_x,chunk_size_y)
    elstop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(path+'s1.h5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=5, mode='auto')
    model.fit(train_x, label_y, epochs=input_epochs, batch_size=input_batch_size,
                    validation_data=(valid_x, valid_y), verbose=2, shuffle=False, callbacks=[elstop,checkpointer,reduce_lr])

    # test_predict = model.predict(test_x)
    # pred_cls = np.argmax(test_predict,axis=1)
    # test_y = np.argmax(test_y,axis=1)

    # kappa = cohen_kappa_score(test_y, pred_cls)
    # print(kappa)
    # acc = accuracy_score(test_y, pred_cls)
    # acc0 = accuracy_score(test_y, np.zeros_like(test_y))
    # print(acc, acc0)
    # recall = metrics.recall_score(test_y, pred_cls, average='macro')
    # print('recall:%f' % recall)

def train_2(path):
    df = pd.read_csv(path + '.csv')
    chunk_size_x = 6
    chunk_size_y = 1
    input_epochs = 100
    input_batch_size = 100

    train_x, label_y, valid_x, valid_y, test_x, test_y, scaler= \
        getdata_2(df,chunk_size_x,chunk_size_y)

    model = lstm_model_2(train_x,chunk_size_y)
    elstop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(path + 's2.h5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=5, mode='auto')
    model.fit(train_x, label_y, epochs=input_epochs, batch_size=input_batch_size,
                    validation_data=(valid_x, valid_y), verbose=2, shuffle=False,callbacks=[elstop,checkpointer,reduce_lr])
    #
    # test_predict = model.predict(test_x)
    # pred_cls = np.argmax(test_predict,axis=1)
    # test_y = np.argmax(test_y,axis=1)

    # kappa = cohen_kappa_score(test_y, pred_cls)
    # print(kappa)
    # acc = accuracy_score(test_y, pred_cls)
    # acc0 = accuracy_score(test_y, np.zeros_like(test_y))
    # print(acc, acc0)
    # recall = metrics.recall_score(test_y, pred_cls, average='macro')
    # print('recall:%f' % recall)

def test_classify(model,path):
    df = pd.read_csv('./testset/' + path + '.csv')
    chunk_size_x = 6
    chunk_size_y = 1

    test_x, test_y, scaler = \
        getdata_test(df, chunk_size_x, chunk_size_y)

    test_predict = model.predict(test_x)
    pred_cls = np.argmax(test_predict,axis=1)

    kappa = cohen_kappa_score(test_y, pred_cls)
    print('kappa:%f'%kappa)
    acc = accuracy_score(test_y, pred_cls)
    print('accuracy:%f'%acc)
    recall = metrics.recall_score(test_y, pred_cls, average='macro')
    print('recall:%f' % recall)

def test_classify_(model):
    df = pd.read_csv('station304_filled.csv')
    chunk_size_x = 6
    chunk_size_y = 1

    train_x, label_y, valid_x, valid_y, test_x, test_y, scaler = \
        getdata_(df, chunk_size_x, chunk_size_y)

    test_predict = model.predict(test_x)
    pred_cls = np.argmax(test_predict,axis=1)
    test_y = np.argmax(test_y,axis=1)


    print((test_y == 0).sum())
    print((test_y == 1).sum())
    print((test_y == 2).sum())
    print((test_y == 3).sum())

    print((pred_cls == 0).sum())
    print((pred_cls == 1).sum())
    print((pred_cls == 2).sum())
    print((pred_cls == 3).sum())
    # plt.plot(test_y)
    # plt.plot(pred_cls)
    # plt.show()
    kappa = cohen_kappa_score(test_y, pred_cls)
    print(kappa)
    acc = accuracy_score(test_y, pred_cls)
    acc0 = accuracy_score(test_y, np.zeros_like(test_y))
    print(acc, acc0)
    recall = metrics.recall_score(test_y, pred_cls, average='macro')
    print('recall:%f' % recall)

def test_classify2(model,model2,path):
    df = pd.read_csv('./testset/' + path + '.csv')
    chunk_size_x = 6
    chunk_size_y = 1

    test_x, test_y, scaler = \
        getdata_test(df, chunk_size_x, chunk_size_y)

    test_predict = model.predict(test_x)
    pred_cls = np.argmax(test_predict,axis=1)
    # test_y = np.argmax(test_y,axis=1)

    idx = pred_cls > 0

    test_x2 = test_x[idx]
    test_predict2 = model2.predict(test_x2)
    pred_cls[idx] = np.argmax(test_predict2, axis=1) + 1

    kappa = cohen_kappa_score(test_y, pred_cls)
    print('kappa:%f'%kappa)
    acc = accuracy_score(test_y, pred_cls)
    print('accuracy:%f'%acc)
    recall = metrics.recall_score(test_y, pred_cls, average='macro')
    print('recall:%f' % recall)




if __name__ == '__main__':
    trainSets = ['station178_filled','station303_filled','station304_filled','station305_filled','station306_filled']
    testSets = ['station1','station2','station3','station4','station5']
    for i in range(5):
        print(trainSets[i]+'start training')
        train(trainSets[i])
        print('LSTM trained')
        train_1(trainSets[i])
        train_2(trainSets[i])
        print('LSTM2state trained')

    #
    # for i in range(5):
    #     print('\n' + trainSets[i]+'start testing')
    #
    #     model = load_model(trainSets[i]+'.h5')
    #     print('LSTM result:')
    #     test_classify(model,testSets[i])
    #
    #     model1 = load_model(trainSets[i]+'s1.h5')
    #     model2 = load_model(trainSets[i]+'s2.h5')
    #     print('LSTM2state result:')
    #     test_classify2(model1,model2,testSets[i])

