import numpy as np

dataset = np.arange(1,11)
dataset = dataset.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
dataset = encoder.fit_transform(dataset)
print(type(dataset))
print(type(dataset.toarray()))
print(dataset)
print(dataset.toarray())
