#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pandas을 pd로 import합니다
import pandas as pd

# Series 데이터입니다
fruits = {"banana": 3, "orange": 2}
print(pd.Series(fruits))


# In[2]:


# Pandas를 pd로 import합니다
import pandas as pd

# DataFrame 데이터입니다
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)


# In[3]:


# Pandas를 pd로 import합니다
import pandas as pd

# Series용 라벨(인덱스)을 작성합니다
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# Series용 데이터를 대입합니다
data = [10, 5, 8, 12, 3]

# Series를 작성합니다
series = pd.Series(data, index=index)

# 딕셔너리 형을 사용하여 DataFrame용 데이터를 작성합니다
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

# DataFrame을 만듭니다
df = pd.DataFrame(data)

print("Series 데이터")
print(series)
print("\n")
print("DataFrame 데이터")
print(df)


# In[4]:


# Pandas를 pd로 import합니다
import pandas as pd

fruits = {"banana": 3, "orange": 2}
print(pd.Series(fruits))


# In[5]:


# Pandas를 pd로 import합니다
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

# index와 data를 포함한 Series를 만들어 series에 대입하세요
series = pd.Series(data, index=index)
print(series)


# In[47]:


import pandas as pd
fruits = {"banana": 3, "orange": 4, "grape": 1, "peach": 5}
series = pd.Series(fruits)
print(series[0:2])


# In[48]:


print(series[["orange", "peach"]])


# In[6]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# 인덱스 참조로 series의 2~4번째의 세 요소를 꺼내 items1에 대입하세요
items1 = series[1:4]

# 인덱스 값을 지정하는 방법으로 "apple", "banana", "kiwifruit"의 인덱스를 가지는 요소를 꺼내 items2에 대입하세요
items2 = series[["apple", "banana", "kiwifruit"]]
print(items1)
print()
print(items2)


# In[7]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# series_values에 series의 데이터를 대입하세요
series_values = series.values
series_values = series.values

# series_index에 series의 인덱스를 대입하세요
series_index = series.index
series_index = series.index

print(series_values)
print(series_index)


# In[8]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index=index)

# 인덱스가 "pineapple" 이고, 데이터가 12인 요소를 series에 추가하세요 
pineapple = pd.Series([12], index=["pineapple"])
# pineapple = pd.Series(12, index="pineapple")
series = series.append(pineapple)
# series = series.append(pd.Series({"pineapple":12}))라도 OK 
print(series)


# In[9]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

# index와 data를 포함한 Series를 작성하여 series에 대입합니다
series = pd.Series(data, index=index)

# 인덱스가 strawberry인 요소를 제거해 series에 대입하세요
series = series.drop("strawberry")

print(series)


# In[53]:


index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

conditions = [True, True, False, False, False]
print(series[conditions])


# In[54]:


print(series[series >= 5])


# In[10]:


index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# series의 요소 중에서 5 이상 10 미만의 요소를 포함하는 Series를 만들어 series에 다시 대입하세요
series = series[series >= 5][series < 10]

print(series)



# In[11]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# series의 인덱스를 알파벳 순으로 정렬해 items1에 대입하세요
items1 = series.sort_index()
item1 = series.sort_index()
# series의 데이터 값을 오름차순으로 정렬해 items2에 대입하세요
items2 = series.sort_values()
items2 = series.sort_values(a)
print(items1)
print()
print(items2)


# In[57]:


data = {"fruits": ["apple", "orange", "banana", "strawberry",
"kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)


# In[12]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

# series1, series2로 DataFrame을 생성하여 df에 대입하세요
df = pd.DataFrame([series1, series2])

# 출력합니다
print(df)


# In[2]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)
df = pd.DataFrame([series1, series2])

# df의 인덱스가 1부터 시작하도록 설정하세요
df.index = [1, 2]

# 출력합니다
print(df)


# In[3]:


data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
series = pd.Series(["mango", 2008, 7], index=["fruits", "year", "time"])

df = df.append(series, ignore_index=True)
print(df)


# In[15]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
data3 = [30, 12, 10, 8, 25, 3]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

# df에 series3을 추가해 df에 다시 대입하세요
index.append("pineapple")
series3 = pd.Series(data3, index=index)
df = pd.DataFrame([series1, series2])

# df에 다시 대입하세요
df = df.append(series3, ignore_index=True)

# 출력합니다
# df와 추가할 Series의 인덱스가 일치하지 않을 때의 동작을 확인합시다
print(df)


# In[62]:


data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)

df["price"] = [150, 120, 100, 300, 150]
print(df)


# In[16]:


import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

new_column = pd.Series([15, 7], index=[0, 1])

# series1, seires2로 DataFrame을 생성합니다
df = pd.DataFrame([series1, series2])

# df에 새로운 열 "mango"를 만들어 new_column의 데이터를 추가하세요
df["mango"] = new_column

# 출력합니다
print(df)


# In[66]:


data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)

print(df)


# In[67]:


df = df.loc[[1,2],["time","year"]]
print(df)


# In[17]:


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
    
# range(시작행수, 종료행수-1)입니다
df.index = range(1, 11)

# loc[]를 사용하여 df의 2~5행(4개의 행)과 "banana", "kiwifruit"의 2열을 포함한 DataFrame을 df에 대입하세요
# 첫번째 행의 인덱스는 1이며, 이후의 인덱스는 정수의 오름차순입니다
df = df.loc[range(2,6),["banana","kiwifruit"]]

print(df)


# In[2]:


import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)


print(df)


# In[3]:


df = df.iloc[[1, 3], [0, 2]]
print(df)


# In[18]:


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# iloc[]를 사용하여 df의 2~5행(4개의 행)과 "banana", "kiwifruit"의 2열을 포함한 DataFrame을 df에 대입하세요
df = df.iloc[range(1,5), [2, 4]] # 슬라이스를 사용하여 df = df.iloc[1:5, [2,4]] 도 가능합니다 

print(df)


# In[6]:


import pandas as pd
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "time": [1, 4, 5, 6, 3],
        "year": [2001, 2002, 2001, 2008, 2006]}
df = pd.DataFrame(data)

# drop()을 이용하여 df의 0,1행을 삭제합니다
df_1 = df.drop(range(0, 2))

# drop()을 이용하여 df의 열 "year"를 제거합니다
df_2 = df.drop("year", axis=1)

print(df_1)
print()
print(df_2)


# In[19]:


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하여 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# drop()을 이용하여 df에서 홀수 인덱스가 붙은 행만을 남겨 df에 대입하세요
df = df.drop(np.arange(2, 11, 2))
# np.arange(2, 11, 2)는 2에서 10까지의 수열을 2의 간격으로 추출한 것입니다
# 2,4,6,8,10이 출력됩니다
# np.arange(2, 11, 3)은 2에서 10까지의 수열을 3의 간격으로 추출한 것입니다

# drop()을 이용하여 df의 열 "strawberry"를 삭제하여 df에 대입하세요
df = df.drop("strawberry", axis=1) 

print(df)


# In[20]:


import pandas as pd
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "time": [1, 4, 5, 6, 3],
        "year": [2001, 2002, 2001, 2008, 2006]}
df = pd.DataFrame(data)
print(df)

# 데이터를 오름차순으로 정렬합니다(인수로 컬럼을 지정)
df = df.sort_values(by="year", ascending = True)
print(df)

# 데이터를 오름차순으로 정렬합니다(인수에 컬럼 리스트를 지정)
df = df.sort_values(by=["time", "year"] , ascending = True)
print(df)


# In[21]:


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# df를 "apple", "orange", "banana", "strawberry" "kiwifruit"의 순으로 오름차순 정렬하세요
# 정렬한 결과로 만들어진 DataFrame을 df에 대입하세요. 첫번째 인수이면 by는 생략 가능합니다
df = df.sort_values(by=columns)

print(df)


# In[17]:


data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df.index % 2 == 0)
print()
print(df[df.index % 2 == 0])


# In[23]:


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# 필터링을 사용하여 df의 "apple" 열이 5 이상이고, "kiwifruit" 열에서 5 이상의 값을 가진 행을 포함한 DataFrame을 df에 대입하세요
df = df.loc[df["apple"] >= 5]
df = df.loc[df["kiwifruit"] >= 5]
#df = df.loc[df["apple"] >= 5][df["kiwifruit"] >= 5] 라도 OK

print(df)


# In[24]:


import pandas as pd
import numpy as np

index = ["growth", "mission", "ishikawa", "pro"]
data = [50, 7, 26, 1]

# Series를 작성하세요
series = pd.Series(data, index=index)

# 인덱스에 알파벳 순으로 정렬한 series를 aidemy에 대입하세요
aidemy = series.sort_index()

# 인덱스가 "tutor"이고 데이터가 30인 요소를 series에 추가하세요
aidemy1 = pd.Series([30], index=["tutor"])
aidemy2 = series.append(aidemy1)

print(aidemy)
print()
print(aidemy2)

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for index in index:
    df[index] = np.random.choice(range(1, 11), 10)

# range(시작행, 종료행-1) 입니다
df.index = range(1, 11)

# loc[]를 사용하여 df의 2~5행(4개의 행)과 "ishikawa"를 포함하는 DataFrame을 aidemy3에 대입하세요
# 첫번째 행의 인덱스는 1이며, 이후의 인덱스는 정수의 오름차순입니다
aidemy3 = df.loc[range(2,6),["ishikawa"]]
print()
print(aidemy3)


# In[ ]:




