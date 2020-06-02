from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten
import numpy as np

y=np.array(list(range(1,6))*2)
print(f"y:{y}")

# #1번답

# y-=1

# from keras.utils import np_utils

# y=np_utils.to_categorical(y)
# print(f"y:{y}")
# print(f"y.shape:{y.shape}")

from sklearn.preprocessing import OneHotEncoder

y=y.reshape(-1,1)

encode=OneHotEncoder()
encode.fit(y)
y=encode.transform(y).toarray()

print(f"y:{y}")
print(f"y.shape:{y.shape}")

np.arg