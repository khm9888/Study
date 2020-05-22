import numpy as np
from keras.models  import Sequential
from keras.layers import Dense, LSTM

a = np.array(range(1,11))

def split(dataset,y_column):
    x_total=list()
    y_total=list()
    for i in range(len(dataset)-y_column+1):
        x=dataset[i:i+y_column]
        y=dataset[i+y_column]
        x_total.append(x)
        y_total.append(y)
    return np.array(x_total),np.array(y_total)

x,y=split(a,5)

print(x)

