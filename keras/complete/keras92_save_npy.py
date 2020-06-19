from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
dataset = load_iris()

print(type(dataset))

x_data = dataset.data
y_data = dataset.target

print(dataset.keys())
print(dataset.feature_names)
x_series = pd.Series(x_data[0],index=dataset.feature_names)
print(x_series.head())

print(type(x_data))
print(type(y_data))

# print(dataset.index)

np.save("./data/iris_x.npy",arr=x_data)
np.save("./data/iris_y.npy",arr=y_data)

x_data_load = np.load("./data/iris_x.npy")
y_data_load = np.load("./data/iris_y.npy")

print(type(x_data_load))
print(type(y_data_load))

print(x_data_load.shape)
print(y_data_load.shape)