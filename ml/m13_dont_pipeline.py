#iris, svc


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.layers import Dense, Input
from sklearn.metrics import r2_score,mean_squared_error as mse,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC,SVC
from keras.utils import np_utils
from sklearn.datasets import load_iris

datasets = load_iris()

print(datasets.keys())

x= datasets.data
y= datasets.target

from sklearn.pipeline import Pipeline

