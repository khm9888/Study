import numpy as np

v= np.array(list(np.random.rand(100)))*10
v= v.astype("int")

print(v)
v=list(v)
print(v.count(1))

