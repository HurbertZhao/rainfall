import pandas as pd
import numpy as np

trainSets = ['station178_filled', 'station303_filled', 'station304_filled', 'station305_filled', 'station306_filled']
testSets = ['station1', 'station2', 'station3', 'station4', 'station5']
for i in range(5):
    df = pd.read_csv(trainSets[i] + '.csv')
    testsize = int(0.9*df.shape[0])
    test = df[testsize:]
    test.to_csv('./testset/'+testSets[i]+'.csv',index=False)