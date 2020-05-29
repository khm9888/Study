from keras.datasets import cifar100#자료가져오기
import matplotlib as plt


(x_train,y_train),(x_test,y_test) = cifar100.load_data()

print(f"x_train[59999]:{x_train[0]}")
print(f"y_train[59999]:{y_train[0]}")

print(f"x_train.shape:{x_train.shape}")
print(f"y_train.shape:{y_train.shape}")
print(f"x_test.shape:{x_test.shape}")
print(f"y_test.shape:{y_test.shape}")