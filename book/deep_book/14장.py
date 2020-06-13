#!/usr/bin/env python
# coding: utf-8

# In[10]:             


import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

# 각 수치가 무엇을 나타내는지 컬럼 헤더로 추가합니다
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols",
"Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

df


# In[13]:


import pandas as pd

# 여기에 해답을 기술하세요
df = pd.read_csv(
"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None)
df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

df


# In[15]:


import csv

# with 문을 사용해 파일을 처리합니다
with open("csv0.csv", "w") as csvfile:
    # writer() 메서드의 인수로 csvfile과 개행(줄바꿈) 코드(\n)를 지정합니다
    writer = csv.writer(csvfile, lineterminator="\n")

    # writerow(리스트) 로 행을 추가합니다
    writer.writerow(["city", "year", "season"])
    writer.writerow(["Nagano", 1998, "winter"])
    writer.writerow(["Sydney", 2000, "summer"])
    writer.writerow(["Salt Lake City", 2002, "winter"])
    writer.writerow(["Athens", 2004, "summer"])
    writer.writerow(["Torino", 2006, "winter"])
    writer.writerow(["Beijing", 2008, "summer"])
    writer.writerow(["Vancouver", 2010, "winter"])
    writer.writerow(["London", 2012, "summer"])
    writer.writerow(["Sochi", 2014, "winter"])
    writer.writerow(["Rio de Janeiro", 2016, "summer"])


# In[16]:


import csv
# 여기에 해답을 기술하세요


# In[17]:


import pandas as pd

data = {"city": ["Nagano", "Sydney", "Salt Lake City", "Athens", "Torino", "Beijing", "Vancouver", "London", "Sochi", "Rio de Janeiro"],
        "year": [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016],
        "season": ["winter", "summer", "winter", "summer", "winter", "summer", "winter", "summer", "winter", "summer"]}

df = pd.DataFrame(data)

df.to_csv("csv1.csv")


# In[19]:


import pandas as pd

data = {"OS": ["Machintosh", "Windows", "Linux"],
        "release": [1984, 1985, 1991],
        "country": ["US", "US", ""]}

# 여기에 해답을 기술하세요
df = pd.DataFrame(data)
df.to_csv("OSlist.csv")


# In[22]:


import pandas as pd
from pandas import Series, DataFrame
attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo", "Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["Hiroshi", "Akiko", "Yuki", "Satoru", "Steeve", "Mituru", "Aoi", "Tarou", "Suguru", "Mitsuo"]}

attri_data_frame1 = DataFrame(attri_data1)
attri_data2 = {"ID": ["107", "109"],
               "city": ["Sendai", "Nagoya"],
               "birth_year": [1994, 1988]}

attri_data_frame2 = DataFrame(attri_data2)

# 여기에 해답을 기술하세요
attri_data_frame1.append(attri_data_frame2).sort_values(by="ID", ascending=True).reset_index(drop=True)

print(attri_data_frame1)
# In[23]:


import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

# 일부 데이터를 일부러 누락시킵니다
sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA 

sample_data_frame


# In[24]:


sample_data_frame.dropna()


# In[25]:


sample_data_frame[[0,1,2]].dropna()


# In[27]:


import numpy as np
from numpy import nan as NA
import pandas as pd
np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

# 여기에 해답을 기술하세요
sample_data_frame[[0, 2]].dropna()


# In[28]:


import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

# 일부 데이터를 일부러 누락시킵니다
sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA


# In[29]:


sample_data_frame.fillna(0)


# In[30]:


sample_data_frame.fillna(method="ffill")


# In[31]:


import numpy as np
from numpy import nan as NA
import pandas as pd
np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[6:, 2] = NA

# 여기에 해답을 기술하세요
sample_data_frame.fillna(method="ffill")


# In[32]:


import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

# 일부 데이터를 일부러 누락시킵니다
sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

#fillna로 NaN 부분에 열의 평균값을 대입합니다
sample_data_frame.fillna(sample_data_frame.mean())


# In[33]:


import numpy as np
from numpy import nan as NA
import pandas as pd
np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[6:, 2] = NA

# 여기에 해답을 기술하세요
sample_data_frame.fillna(sample_data_frame.mean())


# In[35]:


import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
print(df["Alcohol"].mean())


# In[37]:


import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# 여기에 해답을 기술하세요
print(df["Magnesium"].mean())


# In[38]:


import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6], 
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b"]}) 

dupli_data


# In[39]:


dupli_data.duplicated()


# In[40]:


dupli_data.drop_duplicates()


# In[42]:


import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6, 7, 7, 7, 8, 9, 9],
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b", "d", "d", "c", "b", "c", "c"]})

# 여기에 해답을 기술하세요
dupli_data.drop_duplicates()


# In[4]:


import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo", "Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["Hiroshi", "Akiko", "Yuki", "Satoru", "Steeve", "Mituru", "Aoi", "Tarou", "Suguru", "Mitsuo"]}
attri_data_frame1 = DataFrame(attri_data1)

attri_data_frame1


# In[2]:


city_map ={"Tokyo":"Kanto", 
           "Hokkaido":"Hokkaido", 
           "Osaka":"Kansai", 
           "Kyoto":"Kansai"}
city_map


# In[5]:


# 새로운 컬럼으로 region을 추가합니다. 해당 데이터가 없는 경우는 NaN입니다
attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)
attri_data_frame1


# In[10]:


import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo", "Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["Hiroshi", "Akiko", "Yuki", "Satoru", "Steeve", "Mituru", "Aoi", "Tarou", "Suguru", "Mitsuo"]
              }

attri_data_frame1 = DataFrame(attri_data1)

# 여기에 해답을 기술하세요
WE_map = {"Tokyo":"east",
          "Hokkaido":"east",
          "Osaka":"west",
          "Kyoto":"west"}

attri_data_frame1["WE"] = attri_data_frame1["city"].map(WE_map)

attri_data_frame1


# In[11]:


import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo", "Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["Hiroshi", "Akiko", "Yuki", "Satoru", "Steeve", "Mituru", "Aoi", "Tarou", "Suguru", "Mitsuo"]}

attri_data_frame1 = DataFrame(attri_data1)


# In[12]:


# 분할 리스트를 만듭니다
birth_year_bins = [1980, 1985, 1990, 1995, 2000]

# 구간 분할을 실시합니다
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)
birth_year_cut_data


# In[13]:


pd.value_counts(birth_year_cut_data)


# In[14]:


group_names = ["first1980", "second1980", "first1990", "second1990"]
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year,birth_year_bins,labels = group_names)
pd.value_counts(birth_year_cut_data)


# In[15]:


pd.cut(attri_data_frame1.birth_year, 2)


# In[17]:


import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID":[100,101,102,103,104,106,108,110,111,113],
               "city":["Tokyo","Osaka","Kyoto","Hokkaido","Tokyo","Tokyo","Osaka","Kyoto","Hokkaido","Tokyo"],
               "birth_year":[1990,1989,1992,1997,1982,1991,1988,1990,1995,1981],
               "name":["Hiroshi","Akiko","Yuki","Satoru","Steeve","Mituru","Aoi","Tarou","Suguru","Mitsuo"]}

attri_data_frame1 = DataFrame(attri_data1)

# 여기에 해답을 기술하세요
pd.cut(attri_data_frame1.ID, 2)


# In[20]:


import pandas as pd
import numpy as np
from numpy import nan as NA
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

# 각각의 수치가 나타내는 바를 컬럼에 추가합니다
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
            "Magnesium", "Total phenols", "Flavanoids",
            "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
            "OD280/OD315 of diluted wines","Proline"]

# 변수 df의 상위 10행을 변수 df_ten에 대입하여 표시하세요
df_ten = df.head(10)
print(df_ten)

# 데이터의 일부를 누락시킵니다
df_ten.iloc[1,0] = NA
df_ten.iloc[2,3] = NA
df_ten.iloc[4,8] = NA
df_ten.iloc[7,3] = NA
print(df_ten)

# fillna() 메서드로 NaN 부분에 열의 평균값을 대입하세요
df_ten.fillna(df_ten.mean())
print(df_ten)

# "Alcohol" 열의 평균을 출력하세요
print(df_ten["Alcohol"].mean())

# 중복된 행을 제거하세요
df_ten.append(df_ten.loc[3])
df_ten.append(df_ten.loc[6])
df_ten.append(df_ten.loc[9])
df_ten = df_ten.drop_duplicates()
print(df_ten)

# Alcohol 열의 구간 리스트를 작성하세요
alcohol_bins = [0,5,10,15,20,25]
alcoholr_cut_data = pd.cut(df_ten["Alcohol"],alcohol_bins)

# 구간 수를 집계하여 출력하세요
print(pd.value_counts(alcoholr_cut_data))


# In[ ]:




