from keras.datasets import cifar100#자료가져오기
import matplotlib as plt
import os

print(f"\n{__file__}")

bsdir2 = os.path.dirname(__file__)#파일 이름을 이용해서 현재 주소 가져오기
bsdir = "../"+bsdir2

print(f"bsdir2:{bsdir2}")
print(f"bsdir:{bsdir}")


(x_train,y_train),(x_test,y_test) = cifar100.load_data()

print(f"x_train[59999]:{x_train[0]}")
print(f"y_train[59999]:{y_train[0]}")

print(f"x_train.shape:{x_train.shape}")
print(f"y_train.shape:{y_train.shape}")
print(f"x_test.shape:{x_test.shape}")
print(f"y_test.shape:{y_test.shape}")