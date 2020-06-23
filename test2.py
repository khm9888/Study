import numpy as np
from sklearn.model_selection import train_test_split

v= np.array(list(np.random.rand(100)))*10
v= v.astype("int")

print(v)
v=list(v)
print(v.count(1))

