import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None

def get_station():
   df = pd.read_csv(r'./sudeste.csv',nrows = 600000)
   listType = df['wsid'].unique()[:5]
   for id in listType:
      station = df[df['wsid']==id]
      station.to_csv('station' + str(id) + '.csv')

def fill_clean(path):
   df = pd.read_csv(path+'.csv')
   prcp = df['prcp']
   prcp.fillna(0,inplace=True)
   df['prcp'] = prcp
   df = df.drop(columns=['gbrd'])
   wdsp = df['wdsp']
   wdsp.fillna(wdsp.mean(),inplace=True)
   df['wdsp'] = wdsp
   gust = df['gust']
   gust.fillna(gust.mean(), inplace=True)
   df['gust'] = gust
   df = df[df['smax']>0]
   df.to_csv(path+'_filled.csv',index=False)


def getDataSet(data,n_in=1, n_out=1, dropnan=True, numlist=0):
   n_vars = 1 if type(data) is list else data.shape[1]
   numlist = np.array(numlist)
   diff = numlist[1:]- numlist[:-1] - 1
   num = []
   cols = []
   for i in range(diff.shape[0]):
      if diff[i] > 0:
         num.append(i + 1)

   if len(num)>0:
      dataset = series_to_supervised(data[:num[0]], n_in, n_out, dropnan)
      cols.append(dataset)
      if len(num) > 1:
         for i in range(1,len(num)):
            dataset = series_to_supervised(data[num[i-1]:num[i]], n_in, n_out, dropnan)
            cols.append(dataset)

      dataset = series_to_supervised(data[num[-1]:], n_in, n_out, dropnan)
      cols.append(dataset)
      agg = pd.concat(cols)
   else:
      agg = series_to_supervised(data, n_in, n_out, dropnan)

   return agg


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
   n_vars = 1 if type(data) is list else data.shape[1]
   df = pd.DataFrame(data)
   cols, names = [], []
   # input sequence (t-n, ... t-1)
   for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
      names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
   # forecast sequence (t, t+1, ... t+n)
   for i in range(0, n_out):
      cols.append(df.shift(-i))
      if i == 0:
         names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
      else:
         names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
   # put it all together
   agg = pd.concat(cols, axis=1)
   agg.columns = names
   # drop rows with NaN values
   if dropnan:
      agg.dropna(inplace=True)
   return agg

def plot(df):
   i = 1
   values = df.values
   groups = range(1, 16)
   plt.figure()
   for group in groups:
      plt.subplot(len(groups), 1, i)
      plt.plot(values[:, 14 + group])
      plt.title(df.columns[14 + group], y=0.5, loc='right')
      i += 1
   plt.show()

if __name__ == "__main__":
   get_station()
   fill_clean('station178')
   fill_clean('station303')
   fill_clean('station304')
   fill_clean('station305')
   fill_clean('station306')
