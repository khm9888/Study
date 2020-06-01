import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd

dataset=load_boston()

print(dataset.keys())

df=pd.DataFrame(dataset.data, columns=dataset.feature_names)

print(df.head())


(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(type(x_train))



np.save("./data/x_train.npy",arr=x_train)
np.save("./data/x_test.npy",arr=x_test)
np.save("./data/y_train.npy",arr=y_train)
np.save("./data/y_test.npy",arr=y_test)

x_train=np.load("./data/x_train.npy")
x_test=np.load("./data/x_test.npy")
y_train=np.load("./data/y_train.npy")
y_test=np.load("./data/y_test.npy")

print(x_train.shape)
print(y_train.shape)