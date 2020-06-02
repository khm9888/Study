import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPool2D, Input
from sklearn.datasets import load_diabetes
from keras.utils import np_utils
import pandas as pd

#데이터구성
dataset = load_diabetes()

# print(dataset.keys(),"\n",'-'*50)
# #dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename']) 

# [print (f"{i} : {dataset[i]} \n {'-'*50}") for i in list(dataset.keys())[2:-2]]

# column=8
epoch=20


for column in range(3,9):

  '''
  DESCR : .. _diabetes_dataset:

  Diabetes dataset
  ----------------

  Ten baseline variables, age, sex, body mass index, average blood
  pressure, and six blood serum measurements were obtained for each of n =
  442 diabetes patients, as well as the response of interest, a
  quantitative measure of disease progression one year after baseline.

  **Data Set Characteristics:**

    :Number of Instances: 442

    :Number of Attributes: First 10 columns are numeric predictive values

    :Target: Column 11 is a quantitative measure of disease progression one year after baseline

    :Attribute Information:
        - Age
        - Sex
        - Body mass index
        - Average blood pressure
        - S1
        - S2
        - S3
        - S4
        - S5
        - S6

  Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times `n_samples` (i.e. the sum of squares of each column totals 1).

  Source URL:
  https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

  For more information see:
  Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.
  (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)
  '''


  # feature_names : ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

  '''
  age age in years

  sex

  bmi body mass index

  bp average blood pressure

  s1 tc, T-Cells (a type of white blood cells)

  s2 ldl, low-density lipoproteins

  s3 hdl, high-density lipoproteins

  s4 tch, thyroid stimulating hormone

  s5 ltg, lamotrigine

  s6 glu, blood sugar level
  '''


  x=dataset.data
  y=dataset.target

  # print(x)
  # print(y)

  df = pd.DataFrame(x, columns=dataset.feature_names)

  # print(df.head())

  # #dimention 확인
  # print(f"x.shape:{x.shape}")
  # print(f"y.shape:{y.shape}")
  # # x.shape:(442, 10)
  # # y.shape:(442,)


  # print(f"x[0]:{x[0]}")
  # print(f"y[0]:{y[0]}")

  # print(f"x:{x}")
  # print(f"y:{y}")

  #2차원이라 무의미하다.

  # x_train=x_train.reshape(-1,x_train.shape[1])
  # x_test=x_test.reshape(-1,x_test.shape[1])

  # #y값이 다양한 것을 보고 다중분류로 판단했었음
  # from keras.utils import np_utils
  # y = np_utils.to_categorical(y)
  '''
  하지만, 혈중당 수치를 확인하는 걸로 보고, pca 적용후 회귀 분류 적용 예정
  '''



  # scaler를 통해서 나눔
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  x=scaler.fit_transform(x)

  # #scaler 적용후 값 변화 확인 위해서
  # df = pd.DataFrame(x, columns=dataset.feature_names)
  # print(df.head())

  from sklearn.decomposition import PCA

  pca = PCA(n_components=column)
  x=pca.fit_transform(x)

  from sklearn.model_selection import train_test_split as tts
  x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

  # print(f"x_train[0]:{x_train[0]}")
  # print(f"y_train[0]:{y_train[0]}")


  #모델

  input1=Input(shape=(column,))
  dense=Dense(3000,activation="relu")(input1)
  dense=Dense(1,activation="relu")(dense)#회귀모델
  model = Model(inputs=input1,outputs=dense)

  # model.summary()

  #트레이닝

  model.compile(loss="mse", optimizer="adam")
  model.fit(x_train,y_train,batch_size=1,epochs=epoch,validation_split=0.3,verbose=0)

  #테스트

  from sklearn.metrics import mean_squared_error as mse , r2_score

  def rmse(y_test,y_pre):
      return np.sqrt(mse(y_test,y_pre))

  print("-"*40)

  loss = model.evaluate(x_test,y_test,batch_size=1)

  y_pre=model.predict(x_test)
  y_pre=y_pre.reshape(y_pre.shape[0])#print할 때 편히 보기 위해서 백터로 변환

  print(__file__)
  print(f"n_component:{column}")
  print(f"epoch:{epoch}")
  print("="*50)

  print(f"loss:{loss}")
  print(f"rmse:{rmse(y_test,y_pre)}")
  print(f"r2:{r2_score(y_test,y_pre)}")
  print(f"y_test[0:20]:{y_test[0:20]}")
  print(f"y_pre[0:20]:{y_pre[0:20]}")

  #keras79_boston_diabets_dnn

'''
45/45 [==============================] - 0s 1ms/step
keras79_boston_diabets_dnn
n_component:2
epoch:20
==================================================
loss:3735.0725906994608
rmse:61.11523990744036
r2:0.46821917169889526
y_test[0:20]:[306.  67. 270. 135.  92.  42. 215.  72.  63. 321.  81. 136.  96. 111.
  90.  72. 265. 241.  75. 142.]
y_pre[0:20]:[259.27686   97.21536  228.30789  109.3975   125.45214  131.40108
 218.04549  117.35801   76.51561  173.83798  177.48117  108.28813
  87.062584 156.41254  137.89035  130.42747  154.11456  225.1467
  96.73707  107.640144]
----------------------------------------
45/45 [==============================] - 0s 1ms/step
keras79_boston_diabets_dnn
n_component:3
epoch:20
==================================================
loss:3028.3205065409343
rmse:55.03017843337432
r2:0.3819658082596339
y_test[0:20]:[292. 141. 233. 225. 270. 220.  61. 200. 101.  96. 126. 139.  65.  25.
  89.  95.  69.  99. 113. 202.]
y_pre[0:20]:[214.5062   187.68344  182.07265  148.45512  213.55553  230.57115
 117.56571  174.6371    75.259445 135.93633  149.6764   230.6943
  84.32965  128.65097  179.76244  131.52724  170.2981   173.22945
 127.43439  132.9304  ]
----------------------------------------
45/45 [==============================] - 0s 953us/step
keras79_boston_diabets_dnn
n_component:4
epoch:20
==================================================
loss:2605.0366953637863
rmse:51.03956006647835
r2:0.48786958877202125
y_test[0:20]:[129. 144. 225.  96. 232. 140. 131. 160. 101. 141. 147.  98. 311. 122.
  91. 219.  94. 268. 281. 100.]
y_pre[0:20]:[149.62656  169.30319  216.31369  107.95535  185.94403  112.04914
 164.0816   114.97307  170.61102  145.11874  160.72495   86.01105
 183.74176  184.25862  178.78539  136.16042   86.617424 215.07655
 264.83243  168.8797  ]
----------------------------------------
45/45 [==============================] - 0s 909us/step
keras79_boston_diabets_dnn
n_component:5
epoch:20
==================================================
loss:2886.1319387435915
rmse:53.72273183013562
r2:0.5453398459969929
y_test[0:20]:[321. 208.  65. 275. 140.  77.  88. 308. 178.  96.  49. 142. 148. 101.
 120.  70. 116. 293. 184. 156.]
y_pre[0:20]:[226.82043  223.85446   88.50185  217.11685  196.23212  104.019424
 140.66382  245.15468  190.4717   113.28239   85.09343  117.54612
 127.17991   81.75363  192.15543   54.905987  55.3812   196.96996
 169.5107   153.54367 ]
----------------------------------------
45/45 [==============================] - 0s 953us/step
keras79_boston_diabets_dnn
n_component:6
epoch:20
==================================================
loss:2100.5936957889135
rmse:45.832234656625154
r2:0.6700776302113076
y_test[0:20]:[116. 113. 275.  90. 262. 127. 200.  64. 131.  72.  71.  94.  85. 303.
  45.  48.  96. 244. 160. 295.]
y_pre[0:20]:[125.8638   118.10446  223.78249  132.77023  160.19295  162.2146
 159.56932  128.55919  165.32077   77.74411   93.43506  142.00111
  62.92441  249.38515   40.741947  54.33543  103.77912  185.172
 101.4494   202.68431 ]
----------------------------------------
45/45 [==============================] - 0s 909us/step
keras79_boston_diabets_dnn
n_component:7
epoch:20
==================================================
loss:3774.6203545888266
rmse:61.43793919547419
r2:0.3260274506752384
y_test[0:20]:[292.  69. 154. 262.  79. 128. 142. 141. 113. 155. 237. 196. 265. 125.
 146. 110. 200. 276.  84. 109.]
y_pre[0:20]:[195.10179   94.94888  147.89091  145.69044  116.04697  101.06496
 166.92043  143.24284  102.665    233.12573  158.68114  148.33008
 178.62643  100.96037  152.49126  156.16626  110.893616 128.7277
  91.18015  203.098   ]
----------------------------------------
45/45 [==============================] - 0s 953us/step
keras79_boston_diabets_dnn
n_component:8
epoch:20
==================================================
loss:3163.881893487109
rmse:56.24839474977889
r2:0.5030145670507342
y_test[0:20]:[ 85. 132. 163.  55. 191. 216.  72. 172. 233.  51.  72.  90. 248.  59.
  69.  42. 116. 184.  55. 187.]
y_pre[0:20]:[146.7678  255.89783 230.19832  81.06002 178.035   158.0915   63.41456
 149.91959 204.17772  79.99629  73.53751 180.45166 224.72987  74.31783
 120.09038  66.16269  43.62414 167.79695 151.81447 134.21156]
----------------------------------------
45/45 [==============================] - 0s 975us/step
keras79_boston_diabets_dnn
n_component:9
epoch:20
==================================================
loss:2951.573599359724
rmse:54.328386380281756
r2:0.4128748404678919
y_test[0:20]:[177. 137. 265. 155. 173. 281. 275.  59.  39. 244. 182. 103. 129. 101.
  87. 170. 116. 252. 233. 268.]
y_pre[0:20]:[109.463455  88.87362  186.213    140.34972  205.70105  221.87714
 205.73517  121.6142    57.568127 170.54897  126.45313  108.9708
 146.81226   90.1629    74.54963  123.85504  125.78335  134.1372
 181.7995   196.46962 ]
----------------------------------------
45/45 [==============================] - 0s 1ms/step
keras79_boston_diabets_dnn
epoch:20
==================================================
loss:3288.578915479448
rmse:57.34613148085909
r2:0.45440983603916496
y_test[0:20]:[102.  47. 276.  61. 265. 128. 233. 181.  97.  64. 135. 182. 311. 147.
 220. 308. 185. 101.  66. 303.]
y_pre[0:20]:[ 94.635315  51.354637 160.90921  145.60599  210.69516  237.20694
 181.98947  164.0801   152.27464  121.92607  107.68261  119.51425
 177.374    174.99022  171.49948  275.60666  171.22336  107.21431
 172.61519  266.50266 ]
'''