import numpy as np
import pandas as pd

test=pd.read_csv("./test.csv",index_col=0,header=0,encoding='cp949',sep=',')

# print(test)
# print(test.shape)

train=pd.read_csv("./train.csv",index_col=0,header=0,encoding='cp949',sep=",")

# print(train)
# print(train.shape)

gender_submission=pd.read_csv("./gender_submission.csv",index_col=0,header=0,encoding="cp949",sep=',')

# print(gender_submission)
# print(gender_submission.shape)


test_np=test.values
train_np=train.values
gender_submission_np=gender_submission.values

print(f"train.shape:{train.shape}")

print(f"test.shape:{test.shape}")

print(f"gender_submission.shape:{gender_submission.shape}")

print("-----------train info-----------")
print(f'train.info():{train.info()}')
print("-----------test info-----------")
print(f'test.info():{test.info()}')

import matplotlib.pyplot as plt

import seaborn as sns

def 