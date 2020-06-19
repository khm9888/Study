import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPooling2D, Input
from keras.datasets import cifar10

#데이터구성

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train[0])
# print('y_train[0] : ', y_train[0])

# print(f"x_train.shape:{x_train.shape}")
# print(f"y_train.shape:{y_train.shape}")

