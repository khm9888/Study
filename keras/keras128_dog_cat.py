# import tensorflow.k

from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img

img_dog = load_img('./data\dog_cat/dog.jpg', target_size = (224, 224))
img_cat = load_img('./data\dog_cat/cat.jpg', target_size = (224, 224))
img_suit = load_img('./data\dog_cat/suit.jpg', target_size = (224, 224))
img_yangpa = load_img('./data\dog_cat/yangpa.jpg', target_size = (224, 224))

plt.imshow(img_cat)
plt.imshow(img_yangpa)
plt.show()

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yangpa = img_to_array(img_yangpa)

print(arr_dog)
print(type(arr_dog))

#RGB -> BGR
from keras.applications.vgg16 import preprocess_input

arr_cat = preprocess_input(arr_cat)
arr_dog = preprocess_input(arr_dog)
arr_suit = preprocess_input(arr_suit)
arr_yangpa= preprocess_input(arr_yangpa)

print(arr_dog)

import numpy as np

arr_input = np.stack([arr_dog,arr_cat,arr_suit,arr_yangpa])

print(arr_input.shape)

model = VGG16()
probs = model.predict(arr_input)

print(probs)
print("probs.shape")
print(probs.shape)



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