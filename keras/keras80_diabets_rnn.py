import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPool2D, Input,LSTM
from sklearn.datasets import load_diabetes
from keras.utils import np_utils
import pandas as pd

#데이터구성
dataset = load_diabetes()

# print(dataset.keys(),"\n",'-'*50)
# #dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename']) 

# [print (f"{i} : {dataset[i]} \n {'-'*50}") for i in list(dataset.keys())[2:-2]]

# column=8
epoch=10


for column in range(6,8):

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



  # #y값이 다양한 것을 보고 다중분류로 판단했었음
  # from keras.utils import np_utils
  # y = np_utils.to_categorical(y)
  '''
  하지만, 혈중당 수치를 확인하는 걸로 보고, pca 적용후 회귀 분류 적용 예정
  '''



  # scaler를 통해서 255로 나눔
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  x=scaler.fit_transform(x)

  # #scaler 적용후 값 변화 확인 위해서
  # df = pd.DataFrame(x, columns=dataset.feature_names)
  # print(df.head())

  from sklearn.decomposition import PCA

  pca = PCA(n_components=column)
  x=pca.fit_transform(x)

  #2->3 차원으로 변경

  x=x.reshape(-1,x.shape[1],1)


  from sklearn.model_selection import train_test_split as tts
  x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

  # print(f"x_train[0]:{x_train[0]}")
  # print(f"y_train[0]:{y_train[0]}")


  #모델

  input1=Input(shape=(column,1))
  dense=LSTM(3000,activation="relu")(input1)
  dense=Dense(1,activation="relu")(dense)#회귀모델
  model = Model(inputs=input1,outputs=dense)

  # model.summary()

  #트레이닝

  model.compile(loss="mse", optimizer="adam")
  model.fit(x_train,y_train,batch_size=1,epochs=epoch,validation_split=0.3,verbose=1)

  #테스트

  from sklearn.metrics import mean_squared_error as mse , r2_score

  def rmse(y_test,y_pre):
      return np.sqrt(mse(y_test,y_pre))

  print(__file__)

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

  #keras80_boston_diabets_rnn

'''
keras80_boston_diabets_rnn
n_component:6
epoch:10
==================================================
loss:3685.0326041013004
rmse:60.70447012651381
r2:0.3471070897160422
y_test[0:20]:[ 90. 144.  83.  68.  45. 182.  72. 263.  60. 293. 210. 220. 111. 241.
 235. 261. 175. 160. 122.  65.]
y_pre[0:20]:[125.41478  119.929344 111.232254 106.393005  88.47382  122.57013
  82.2123   183.51384   99.66615  137.57268  107.31585  143.8481
 126.911804 193.69319  141.17422  186.3622   134.96504  106.21417
 151.13809  101.46578 ]
 ---------------------------------------
 

keras80_boston_diabets_rnn
n_component:7
epoch:10
==================================================
loss:3451.027596860462
rmse:58.745446697070925
r2:0.3665254164691826
y_test[0:20]:[246. 215.  59. 262. 230. 175.  77. 248. 135. 135. 210.  97. 178. 263.
 279. 150.  60. 232. 109.  55.]
y_pre[0:20]:[196.05212  188.59045   97.640594 178.69244  262.22073  151.93033
 100.065865 194.50436  120.42812  132.39397  107.126434  86.221565
 122.05198  180.31973  166.48419  150.44287   96.56704  196.20537

'''
