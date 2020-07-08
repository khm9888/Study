import numpy as np
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

img_dog = load_img('./data/dog_cat/dog.jpg', target_size = (224, 224))
img_cat = load_img('./data/dog_cat/cat.jpg', target_size = (224, 224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size = (224, 224))
img_yangpa = load_img('./data/dog_cat/yangpa.jpg', target_size = (224, 224))

plt.imshow(img_yangpa)
plt.imshow(img_dog)
# plt.show()

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yangpa = img_to_array(img_yangpa)

# print("dog :", arr_dog)
'''
전처리 전
  ...
  [ 85.  89. 101.]
  [ 84.  88. 100.]
  [ 91.  97. 109.]]] -> 갈수록 커짐
'''
print("dog_type :",type(arr_dog))  # dog_type : <class 'numpy.ndarray'>
print("dog_shape :", arr_dog.shape) # dog_shape : (224, 224, 3)

print("=====================================================")

# vgg16 모델 사용하기 전 전처리를 해 준다

# RGB -> BGR (vgg16의 전처리 방법)
from keras.applications.vgg16 import preprocess_input

# 우리가 가져온 이미지들을
# vgg16 model에 잘 넣기 위해서 하는 전처리방법이다

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yangpa = preprocess_input(arr_yangpa)


# print("arr_dog :", arr_dog)
print("arr_dog.shape :", arr_dog.shape) # (224, 224, 3)
'''
전처리 후
  ...
  [  -2.939003   -27.779      -38.68     ]
  [  -3.939003   -28.779      -39.68     ]
  [   5.060997   -19.779      -32.68     ]]] -> 갈수록 작아짐 / 전처리 후 반대가 됨 RGB -> BGR
'''

# 이미지 데이터를 하나로 합친다
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yangpa]) # (4, 224, 224, 3) 형태가 될 것
# 확인
print("arr_input.shape :", arr_input.shape) # (4, 224, 224, 3)

# 2. 모델 구성
model = VGG16()
probs = model.predict(arr_input)

print("probs :", probs)
# probs : [[8.1733988e-09 3.4492298e-09 5.1760191e-10 ... 1.1114131e-09
#   1.3721666e-08 2.1572794e-07]
#  [1.0318995e-06 4.3254713e-06 1.6692758e-06 ... 8.0409114e-07
#   7.9598732e-04 2.6367083e-05]
#  [9.0286022e-07 1.7122602e-07 1.6753944e-07 ... 5.2525511e-08
#   2.4428475e-06 1.1885550e-05]
#  [1.0904158e-06 2.2400548e-06 2.0820648e-06 ... 1.0274141e-06
#   3.7983162e-05 1.8630184e-04]]
print("probs.shape :", probs.shape) # (4, 1000)

# 이미지 결과
from keras.applications.vgg16 import decode_predictions # decode_predictions : 예측값을 해석해준다 / vgg16에서 제공하는 기능

results = decode_predictions(probs)
print("dog? :", results[0])
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("cat? :", results[1])
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("suit? :", results[2])
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("yangpapa? :", results[3])