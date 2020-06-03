import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sam_df=pd.read_csv("./data/csv/samsung.csv",encoding="cp949",index_col=0,header=0)

sam_df=sam_df[:509]

for i in range(len(sam_df.index)):
    if type(sam_df.iloc[i,0])==str:
        sam_df.iloc[i,0] = int(sam_df.iloc[i,0].replace(",",""))

sam=sam_df.values

def split(sam,count):
    xs=list()
    ys=list()
    for i in range(len(sam)-count+1):
        x=sam[i:i+count]
        y=sam[i+count]
        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys)

# split(sam,5)

from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler

stand=StandardScaler()
sam_stand=stand.fit_transform(sam)

robust=RobustScaler()
sam_robust=robust.fit_transform(sam)

minmax=MinMaxScaler()
sam_minmax=minmax.fit_transform(sam)

plt.scatter(sam_stand,sam_robust)
plt.show()


plt.hist(sam_robust)
plt.show()



plt.hist(sam_minmax)
plt.show()


