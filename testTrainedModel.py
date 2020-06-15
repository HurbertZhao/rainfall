from keras.models import load_model
from LSTM_RAINFALL import test_classify,test_classify2
from SVR_C import test as SVMtest
from RandomForest import test as RFtest
from xg import test as XGtest
from Adaboost import test as AdaBtest
from bagging import vote
from sklearn.externals import joblib

if __name__ == '__main__':

    trainSets = ['station178_filled','station303_filled','station304_filled','station305_filled','station306_filled']
    testSets = ['station1','station2','station3','station4','station5']

    # LSTM
    for i in range(5):
        print('\n' + trainSets[i]+'start testing')

        model = load_model(trainSets[i]+'.h5')
        print('LSTM result:')
        test_classify(model,testSets[i])

        model1 = load_model(trainSets[i]+'s1.h5')
        model2 = load_model(trainSets[i]+'s2.h5')
        print('LSTM2state result:')
        test_classify2(model1,model2,testSets[i])

    # SVM
    for i in range(5):
        print(trainSets[i]+' start testing')
        model = joblib.load(trainSets[i] + 'SVM.pkl')
        print('SVM result:')
        SVMtest(model,testSets[i])

    # RandomForest
    for i in range(5):
        print('\n' + trainSets[i]+' start testing')
        model = joblib.load(trainSets[i] + 'RF.pkl')
        print('RandomForest result:')
        RFtest(model,testSets[i])

    # XGBoost
    for i in range(5):
        print('\n' + trainSets[i] + ' start testing')
        model = joblib.load(trainSets[i] + ".joblib.dat")
        print('XGBOOST result:')
        XGtest(model,testSets[i])

    # bagging
    for i in range(5):
        print('\n' + trainSets[i]+' start testing')
        vote(trainSets[i],testSets[i])
        print('Bagging result:')


    #Adaboost
    for i in range(5):
        print('\n' + trainSets[i]+' start testing')
        model = joblib.load(trainSets[i] + '10.joblib')
        print('10 estimators result:')
        AdaBtest(model, testSets[i])
        print('20 estimators result:')
        model1 = joblib.load(trainSets[i] + '20.joblib')
        AdaBtest(model1, testSets[i])

