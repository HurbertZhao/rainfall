# README
## 文件作用
data_operate.py对数据集进行处理，提取出前5个气象站的数据，并对数据进行清洗和填充.<br>
testset文件夹含有station1-station5的5个气象站的数据构成的测试集，可以使用gettestset.py得到，使用该命令前请确保已经使用过data_operate.py得到处理过的数据集.<br>
LSTM_RAINFALL.py文件为训练LSTM模型的文件，其中train()函数是训练LSTM模型，train_1,train_2分别训练2模型LSTM的第一个和第二个模型，如果训练时间过长可以进入LSTM_RAINFALL.py文件更改函数中的epoch，但是这样会影响最终的测试结果.<br>
SVC_R.py是训练SVM模型的文件.<br>
RandomForest.py是训练随机森林模型的文件.<br>
xg.py是训练XGBoost模型的文件.<br>
Adaboost.py是训练Adaboost模型，其中若分类器是决策树模型.<br>
bagging是训练bagging模型，其中若分类器是LSTM模型，由于本次任务对每个气象站分别建模，而每个气象站会训练10个LSTM弱分类器模型，则一共需要训练50个LSTM模型，因此该文件训练时间会非常长，还望助教能耐心等待.<br>
testTrainModel.py对已经训练好的上述4个模型和2中ensemble方法进行测试，使用的测试集为testset中的5个，在使用前请确保上述模型均已训练完成，如果有没有训练好的情况可以进入该文件将对应模型部分注释掉，文件中每种摸得的部分都使用了注释进行标注.<br>
testBestModel.py是对提交的最好的模型进行测试的文件.<br>
report.pdf是我们所撰写的报告文件


## 软件环境和版本
本次实验我们使用操作系统为Windows10，语言为Python3.8，所需的库有：<br>
pandas 1.0.1 
```shell 
pip install pandas
```
numpy 1.18.0 
```shell
pip install numpy
```
torch 1.4.0
```shell
pip install Pytorch
```
Keras 2.3.1 
```shell
pip install keras
```
tensorflow 2.2.0 
```shell
pip install tensorflow
```
xgboost 1.1.0
```shell
pip install xgboost
```
scikit-learn 0.22.2.post1 
```shell
pip install sklearn
```
## 下载数据及整理数据
数据集下载链接：https://cloud.tsinghua.edu.cn/d/a96c9fb8f56d4fb5be62/<br>
下载之后请解压，并直接将csv文件移动至到项目codes目录下<br>
数据集整理：<br>
```shell
data_operate.py
```
注意：必须进行数据集整理，否则后面无法进行。
## 训练模型
训练LSTM模型和2模型LSTM模型：<br>
```shell
LSTM_RAINFALL.py
```
训练SVM模型：<br>
```shell
SVR_C.py
```
训练随机森林模型：<br>
```shell
RandomForest.py
```
训练XGBoost模型:<br>
```shell
xg.py
```
训练bagging：<br>
```shell
bagging.py
```
训练Adaboost：<br>
```shell
Adaboost.py
```

## 训练模型测试
```shell
testTrainedModel.py
```
## 最优模型测试
```shell
trainBestModel.py
```
