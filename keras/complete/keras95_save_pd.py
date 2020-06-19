import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset=pd.read_csv("./data/csv/iris.csv",index_col=0,header=0,sep=",")

# print(dataset.head())
# print(dataset.tail())

# print(dataset.values)
# print(type(dataset.values))

# print(list(dataset.keys))
# print(np.array(list(dataset.keys())))
# df1=pd.DataFrame(dataset.values(),columns=np.array(list(dataset.keys())))


np.save("./data/1.npy",arr=dataset.values)
x=np.load("./data/1.npy")

print(x)
print(x.shape)

df = pd.DataFrame(x,columns=dataset.keys())

print(df)

# x_train = dataset.values[1:]
# print(x_train)

# np.save("./data/iris_x.npy", arr=x_train)
# x_train_load=np.load("./data/iris_x.npy")

# print(x_train.shape)

