#part11

#kospi200 데이터를 이용한 삼성전자 주가예측

#삼성전자 주식 가격과 kospi 가격을 이용해서, 내일의 삼성전자 주가 예측

#삼성전자 데이터를 가지고 DNN / RNN 모델을 만들고 비교

#그 후 입력 데이터 둘(삼성전자, kospi200)을 DNN과 :LSTM 로 구성하고, 앙상블 다:1로 비교




#저장된 numpy 불러오기


import numpy as np
import pandas as pd


#allow_pickle=True, 적용하면 된다.
kospi200=np.load("C:\\Users\\bitcamp\Desktop\part10\kospi200.npy",allow_pickle=True)
samsung=np.load("c:\\Users\\bitcamp\Desktop\part10\samsung.npy",allow_pickle=True)

print(kospi200)
print(samsung)

print(kospi200.shape)
print(samsung.shape)
