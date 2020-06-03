#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
get_ipython().run_line_magic('matplotlib', 'inline')
def aidemy_imshow(name, img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    plt.imshow(img)
    plt.show()

cv2.imshow = aidemy_imshow


# In[2]:


# 미리 "cleansing_data" 폴더를 실행 파일과 같은 폴더(주피터 노트북 소스코드가 저장된 폴더)에 작성하여 15장의 샘플 파일인 "sample.jpg"를 넣어주세요

# import합니다
import numpy as np
import cv2

# 이미지를 읽습니다
# "cleansing_data" 폴더에 sample.jpg가 존재할 때의 코드입니다
img = cv2.imread("cleansing_data/sample.jpg")

# sample은 윈도우의 이름입니다
cv2.imshow("sample", img)


# In[3]:


import numpy as np
import cv2

# 여기에 해답을 기술하세요
# OpenCV를 사용하여 이미지를 읽습니다
img = cv2.imread("cleansing_data/sample.jpg")

# 이미지를 출력합니다
cv2.imshow("sample", img)


# In[4]:


import numpy as np
import cv2

# 이미지의 크기를 결정합니다
img_size = (512, 512)

# 이미지 정보를 가지는 행렬을 만듭니다
# 빨간색 이미지이므로, 각 요소가 [0, 0, 255]인 512x512의 행렬을 만듭니다

# 행렬이 전치되는 것에 주의합니다
# 이미지 데이터의 각 요소는 0~255의 값만 지정 가능합니다. 이를 명시하기 위해 dtype 옵션으로 데이터의 형식을 결정합니다


my_img = np.array([[[0, 0, 255] for _ in range(img_size[1])] for _ in range(img_size[0])], dtype="uint8")

# 표시합니다
cv2.imshow("sample", my_img)

# 저장합니다
# 파일명 my_img.jpg
cv2.imwrite("my_red_img.jpg", my_img)


# In[5]:


import numpy as np
import cv2

# 이미지의 크기를 결정합니다
img_size = (512, 512)

# 512x512 크기의 녹색 이미지를 만드세요
img = np.array([[[0, 255, 0] for _ in range(img_size[1])] for _ in range(img_size[0])], dtype="uint8")

cv2.imshow("sample", img)


# In[6]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")
size = img.shape

# 이미지를 나타내는 행렬의 일부를 꺼내면, 그것이 트리밍이 됩니다
# n등분을 하려면 가로세로 크기를 나눕니다.
my_img = img[: size[0] // 2, : size[1] // 3]

# 여기에서는 원래의 배율을 유지하면서 폭과 높이를 각각 2배로 합니다. 크기를 지정할 때는 (폭, 높이)의 순서라는 점을 유의하세요
my_img = cv2.resize(my_img, (my_img.shape[1] * 2, my_img.shape[0] * 2))

cv2.imshow("sample", my_img)


# In[7]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 여기에 해답을 기술하세요
my_img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))

cv2.imshow("sample", my_img)


# In[8]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# warpAffine() 함수 사용에 필요한 행렬을 만듭니다
# 첫번째 인수는 회전의 중심입니다(여기에서는 이미지의 중심을 설정)
# 두번째 인수는 회전의 각도입니다(여기에서는 180도를 설정)
# 세번째 인수는 배율입니다(여기에서는 2배 확대로 설정)
mat = cv2.getRotationMatrix2D(tuple(np.array(img.shape[:2]) / 2), 180, 2.0)

# 아핀 변환을 합니다
# 첫번째 인수는 변환하려는 이미지입니다
# 두번째 인수는 위에서 생성한 행렬(mat)입니다
# 세번째 인수는 사이즈입니다

my_img = cv2.warpAffine(img, mat, img.shape[:2])

cv2.imshow("sample", my_img)


# In[9]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 여기에 해답을 기술하세요
my_img = cv2.flip(img, 0)

cv2.imshow("sample", my_img)


# In[2]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 색 공간을 변환합니다
my_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

cv2.imshow("sample", my_img)


# In[3]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 여기에 해답을 기술하세요
for i in range(len(img)):
    for j in range(len(img[i])):
        for k in range(len(img[i][j])):
            img[i][j][k] = 255 - img[i][j][k]

cv2.imshow("sample", img)


# In[4]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 첫번째 인수가 처리하는 이미지입니다
# 두번째 인수가 임계값입니다
# 세번째 인수가 최대값(maxvalue)입니다
# 네번째 인수는 THRESH_BINARY, THRESH_BINARY_INV, THRESH_TOZERO, THRESH_TRUNC, THRESH_TOZERO_INV 중 하나입니다. 각각의 설명은 다음과 같습니다

#THRESH_BINARY: 픽셀값이 임계값을 초과하는 경우 해당 픽셀을 maxValue로 하고, 그 이외의 경우 0(검은색)으로 합니다
#THRESH_BINARY_INV: 픽셀값이 임계값을 초과하는 경우 0으로 설정하고, 그 이외의 경우 maxValue로 합니다

#THRESH_TRUNC: 픽셀값이 임계값을 초과하는 경우 임계값으로 설정하고, 그 이외의 픽셀은 변경하지 않습니다

#THRESH_TOZERO: 픽셀값이 임계값을 초과하는 경우 변경하지 않고, 그 이외의 경우 0으로 설정합니다
#THRESH_TOZERO_INV: 픽셀값이 임계값을 초과하는 경우 0으로 설정하고, 그 이외의 경우 변경하지 않습니다

# 임계값을 75로, 최대값을 255로 하여, THRESH_TOZERO 를 적용합니다
# 임계값도 반환되므로 retval으로 돌려받습니다
retval, my_img = cv2.threshold(img, 75, 255, cv2.THRESH_TOZERO)

cv2.imshow("sample", my_img)


# In[5]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 여기에 해답을 기술하세요
retval, my_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("sample", my_img)


# In[6]:


# 미리 "cleansing_data" 폴더에 15장 샘플 "mask.png" 파일을 넣어 두세요
import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 두번째 인수로 0을 지정하면 채널수가 1인 이미지로 변환해 읽습니다
mask = cv2.imread("cleansing_data/mask.png", 0)

# 원래 이미지와 같은 크기로 리사이즈합니다
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# 세번째 인수로 마스크용 화상을 선택합니다
my_img = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow("sample", my_img)


# In[7]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")
mask = cv2.imread("cleansing_data/mask.png", 0)
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# 여기에 해답을 기술하세요
retval, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
my_img = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow("sample", my_img)


# In[8]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 첫번째 인수는 원본 이미지입니다
# 두번째는 n x n(마스크 크기)에서 n값을 지정합니다(n은 홀수)
# 세번째 인수는 x축 방향의 편차(일반적으로 0 지정)입니다
my_img = cv2.GaussianBlur(img, (5, 5), 0)

cv2.imshow("sample", my_img)


# In[9]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 여기에 해답을 기술하세요
my_img = cv2.GaussianBlur(img, (21, 21), 0)

cv2.imshow("sample", my_img)


# In[10]:


import numpy as np
import cv2
  
img = cv2.imread("cleansing_data/sample.jpg")
my_img = cv2.fastNlMeansDenoisingColored(img)

cv2.imshow("sample", my_img)


# In[11]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cleansing_data/sample.jpg")

# 필터의 정의
filt = np.array([[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]], np.uint8)

# 팽창 처리합니다
my_img = cv2.dilate(img, filt)

cv2.imshow("sample", my_img)


# In[12]:


import numpy as np
import cv2

img = cv2.imread("cleansing_data/sample.jpg")

# 여기에 해답을 기술하세요
filt = np.array([[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]], np.uint8)

# 침식 처리합니다
my_img = cv2.erode(img, filt)
cv2.imshow("sample", my_img)

# 비교하기 위해 원본 사진을 표시합니다
cv2.imshow("original", img)
plt.show()


# In[13]:


import cv2
import numpy as np

img = cv2.imread("cleansing_data/sample.jpg")

# 원본 이미지를 지정합니다
cv2.imshow('Original', img)

# 흐림 처리를 구현하세요(두번째 인수에는 77,77을 지정하세요)
blur_img = cv2.GaussianBlur(img, (77,77), 0)
cv2.imshow('Blur', blur_img)

# 이미지의 색상을 반전시키세요
bit_img = cv2.bitwise_not(img)
cv2.imshow('Bit', bit_img)

# 임계값 처리를 하세요(임계값을 90으로 하여, 그 이하이면 변경하지 않고, 그 이상이면 0으로 하세요)
retval, thre_img = cv2.threshold(img, 90, 255, cv2.THRESH_TOZERO)
cv2.imshow('THRESH', thre_img)


# In[1]:


#def scratch_image(img, flip=True, thr=True, filt=True, resize=True, erode=True):
    #flip은 이미지의 좌우 반전
    #thr은 임계값 처리
    #filt은 흐림 효과
    #resize는 모자이크 처리
    #erode은 침식 여부를 지정
    #img의 형식은 OpenCV의 cv2.read()로 읽은 이미지 데이터 형입니다. 부풀려진 이미지 데이터를 배열로 반환합니다


# In[15]:


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def scratch_image(img, flip=True, thr=True, filt=True, resize=True, erode=True):
    # ---------------------- 여기에 기술합니다 ---------------------
    # 부풀리기에 사용할 방법을 배열에 정리합니다
    methods = [flip, thr, filt, resize, erode]

    # 이미지의 크기를 취득해 흐림 효과에 사용되는 필터를 만듭니다
    img_size = img.shape
    filter1 = np.ones((3, 3))

    # 원본 이미지 데이터를 배열에 저장합니다
    images = [img]

    # 부풀리기에 이용하는 함수입니다
    scratch = np.array([
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        lambda x: cv2.resize(cv2.resize(
        x, (img_size[1] // 5, img_size[0] // 5)
        ),(img_size[1], img_size[0])),
        lambda x: cv2.erode(x, filter1)
    ])

    # 함수와 이미지를 인수로 받아, 가공된 이미지를 부풀리는 함수입니다
    doubling_images = lambda f, imag: np.r_[imag, [f(i) for i in imag]]

    # methods가 True인 함수로 부풀리기를 실시합니다
    for func in scratch[methods]:
        images = doubling_images(func, images)
        
    return images
    # ---------------------- 여기까지 기술하세요 ---------------------

# 이미지를 읽습니다
cat_img = cv2.imread("cleansing_data/cat_sample.jpg")

# 이미지 데이터를 부풀립니다
scratch_cat_images = scratch_image(cat_img)

# 이미지를 저장할 폴더를 만듭니다
if not os.path.exists("scratch_images"):
    os.mkdir("scratch_images")

for num, im in enumerate(scratch_cat_images):
    # 우선 대상 폴더 "scratch_images/"를 지정하고 번호를 붙여 저장합니다
    cv2.imwrite("scratch_images/" + str(num) + ".jpg" ,im)


# In[17]:


# 함수를 저장합니다
sc_flip = [
    lambda x: x,
    lambda x: cv2.flip(x, 1)
]
sc_thr = [
    lambda x: x,
    lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1]
]
sc_filter = [
    lambda x: x,
    lambda x: cv2.GaussianBlur(x, (5, 5), 0)
]
sc_mosaic = [
    lambda x: x,
    lambda x: cv2.resize(cv2.resize(
        x, (img_size[1] // 5, img_size[0] // 5)
        ),(img_size[1], img_size[0]))
]
sc_erode = [
    lambda x: x,
    lambda x: cv2.erode(x, filter1)
]
# 부풀리기 구현을 한 줄로 정리할 수 있습니다
#[e(d(c(b(a(img))))) for a in sc_flip for b in sc_thr for c in sc_filter for d in sc_mosaic for e in sc_erode]


# In[18]:


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def scratch_image(img, flip=True, thr=True, filt=True, resize=True, erode=True):
    # ---------------------- 여기에 기술합니다 ---------------------
    # 부풀리기에 사용할 방법을 배열에 정리합니다
    methods = [flip, thr, filt, resize, erode]

    # 이미지의 크기를 취득해 흐림 효과에 사용되는 필터를 만듭니다
    img_size = img.shape
    filter1 = np.ones((3, 3))

    # 부풀리기에 이용하는 함수입니다
    scratch = np.array([
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        lambda x: cv2.resize(cv2.resize(
            x, (img_size[1] // 5, img_size[0] // 5)
            ),(img_size[1], img_size[0])),
        lambda x: cv2.erode(x, filter1)
    ])
    act_scratch = scratch[methods]

    # 메서드를 준비합니다
    act_num = np.sum([methods])
    form = "0" + str(act_num) + "b"
    cf = np.array([list(format(i, form)) for i in range(2**act_num)])

    # 이미지 변환 작업을 수행합니다
    images = []
    for i in range(2**act_num):
        im = img
        for func in act_scratch[cf[i]=="1"]: # bool 인덱스를 참조합니다
            im = func(im)
        images.append(im)
    return images
    # ---------------------- 여기까지 기술하세요 ---------------------

# 이미지를 읽습니다
cat_img = cv2.imread("cleansing_data/cat_sample.jpg")

# 이미지 데이터를 부풀립니다
scratch_cat_images = scratch_image(cat_img)

# 이미지를 저장할 폴더를 만듭니다
if not os.path.exists("scratch_images"):
    os.mkdir("scratch_images")

for num, im in enumerate(scratch_cat_images):
    # 우선 대상 폴더 "scratch_images/"를 지정하고 번호를 붙여 저장합니다
    cv2.imwrite("scratch_images/" + str(num) + ".jpg" ,im)


# In[ ]:




