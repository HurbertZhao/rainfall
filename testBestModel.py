from LSTM_RAINFALL import test_classify,test_classify2
from keras.models import load_model
trainSets = ['station178_filled', 'station303_filled', 'station304_filled', 'station305_filled', 'station306_filled']
testSets = ['station1', 'station2', 'station3', 'station4', 'station5']
for i in range(5):
    print('\n' + trainSets[i] + 'start testing')
    model1 = load_model(trainSets[i] + 'bests1.h5')
    model2 = load_model(trainSets[i] + 'bests2.h5')
    print('LSTM2state result:')
    test_classify2(model1, model2, testSets[i])