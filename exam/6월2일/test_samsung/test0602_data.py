import numpy as np
import pandas as pd

samsung = pd.read_csv("./data/csv/samsung.csv",index_col=0,header=0,encoding="cp949",sep=",")
hite = pd.read_csv("./data/csv/hite.csv",index_col=0,header=0,encoding="cp949")

samsung = samsung.dropna(axis=0)#pandas에서는 행
hite = hite.fillna(method="bfill")#pandas에서는 행
print(samsung.shape)
print(hite.head())

# hite.iloc[0,1:5]=[10,20,30,40]
# hite.loc["2020-06-02","고가":"거래량"]=[20,30,40,50]

print(hite.head())
print(hite.describe())

hite=hite[:509]
samsung=samsung[:509]

samsung=samsung.sort_values(["일자"])
hite=hite.sort_values(["일자"])

print(hite.head())
# print(len(samsung))
# samsung.astype(str)
for i in range(len(samsung)):
    samsung.iloc[i] = samsung.iloc[i].replace(',', '')
    # if type(samsung.iloc[i,0])==str:
    #     print(i)

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        try:
            hite.iloc[i,j] = int(hite.iloc[i,j].replace(",",""))
        except:
            print(i)
            
from sklearn.decomposition import PCA

samsung=samsung.values
hite=hite.values

np.save("./data/samsung.npy",arr=samsung)
np.save("./data/hite.npy",arr=hite)

np.load("./data/samsung.npy",allow_pickle=True)