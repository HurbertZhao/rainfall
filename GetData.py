import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_operate import getDataSet

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
