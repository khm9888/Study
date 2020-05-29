import numpy as np
import pandas as pd

df1 = pd.read_csv("./part10/kospi200.csv",index_col=0,header=0,encoding="cp949",sep=",")

df2= pd.read_csv("./part10/kospi200.csv",index_col=0,header=0,encoding="cp949",sep=",")

len(len(df1.index()))

# , 제거
for i in range(len(df1.index)):
    df1.iloc[i,4]=int(df1.iloc[i,4].replace(",",""))
    
for i in range(len(df2.index)):
    for j in range(len(df2.index)):
        df2.iloc[i,j]=int(df1.iloc[i,j].replace(",",""))
        
df1 = df1.sort_values(["일자"])
df2 = df2.sort_values(["일자"])

