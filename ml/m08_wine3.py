#wine 퀄 classify 70 가장 우측이 wine quality
#wine 퀄 classify 70 가장 우측이 wine quality


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Dense, Input,LSTM
from sklearn.metrics import r2_score,mean_squared_error as mse,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC,SVC
from keras.utils import np_utils
from sklearn.decomposition import PCA
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


#0. 데이터 전처리

# wine_df = pd.read_csv("./data/winequality-white.csv",header=0,encoding="cp949",sep=";")
wine_df = pd.read_csv("D:\Study\ml\winequality-white.csv",header=0,encoding="cp949",sep=";")

count_data = wine_df.groupby("quality")["quality"]


print(count_data)

# plt.plot(count_data)
# plt.show()