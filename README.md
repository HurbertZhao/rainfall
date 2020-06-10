# README
## 文件作用

## 软件环境和版本
本次实验我们使用操作系统为Windows10，语言为Python3.8，所需的库有：<br>
pandas 1.0.1 pip install pandas<br>
numpy 1.18.0 pip install numpy<br>
torch 1.4.0 pip install Pytorch<br>
Keras 2.3.1 pip install keras<br>
tensorflow 2.2.0 pip install tensorflow<br>
xgboost 1.1.0 pip install xgboost<br>
scikit-learn 0.22.2.post1 pip install sklearn<br>
## 下载数据及整理数据
数据集下载链接：https://cloud.tsinghua.edu.cn/d/a96c9fb8f56d4fb5be62/<br>
下载之后请解压到项目根目录下<br>
数据集整理：<br>
data_operate.py<br>
## 训练模型
训练LSTM模型和2模型LSTM模型：<br>
LSTM_RAINFALL.py<br>
训练SVM模型：<br>
SVR_C.py<br>
训练随机森林模型：<br>
RandomForest.py<br>
训练XGBoost模型:<br>
xg.py<br>
训练bagging：<br>
bagging.py<br>
训练Adaboost：<br>


## 训练模型测试
testTrainedModel.py<br>
## 最优模型测试
trainBestModel.py
