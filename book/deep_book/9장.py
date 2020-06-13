#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

# 지정된 인덱스와 컬럼을 가진 DataFrame을 난수로 생성하는 함수입니다
def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1,101),len(index))
    df.index = index
    return df

# 인덱스와 컬럼이 일치하는 DataFrame을 만듭니다
columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

# df_data1과 df_data2를 세로로 연결하여 df1에 대입하세요
df1 = pd.concat([df_data1, df_data2], axis=0)

# df_data1과 df_data2를 가로로 연결하여 df2에 대입하세요
df2 = pd.concat([df_data1, df_data2], axis=1)

print(df1)
print(df2)


# In[5]:


import numpy as np
import pandas as pd

# 지정된 인덱스와 컬럼을 가진 DataFrame을 난수로 생성하는 함수입니다
def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df


columns1 = ["apple", "orange", "banana"]
columns2 = ["orange", "kiwifruit", "banana"]

# 인덱스가 1,2,3,4이고 컬럼이 columns1인 DataFrame을 만듭니다
df_data1 = make_random_df(range(1, 5), columns1, 0)

# 인덱스가 1,3,5,7이고 컬럼이 columns2인 DataFrame을 만듭니다
df_data2 = make_random_df(np.arange(1, 8, 2), columns2, 1)

# df_data1과 df_data2를 세로로 연결하여 df1에 대입하세요
df1 = pd.concat([df_data1, df_data2], axis=0)

# df_data1과 df_data2를 가로로 연결하여 df2에 대입하세요
df2 = pd.concat([df_data1, df_data2], axis=1) 

print(df1)
print(df2)


# In[6]:


import numpy as np
import pandas as pd

# 지정된 인덱스와 컬럼을 가진 DataFrame을 난수로 생성하는 함수입니다
def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
            df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

# df_data1과 df_data2를 가로로 연결하고, keys로 "X", "Y"를 지정하여 MultiIndex로 만든 뒤 df에 대입하세요
df = pd.concat([df_data1, df_data2], axis=1, keys=["X", "Y"])

# df의 "Y" 라벨 "banana"를 Y_banana에 대입하세요
Y_banana = df["Y", "banana"]

print(df)
print()
print(Y_banana)


# In[7]:


import numpy as np
import pandas as pd

data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year": [2001, 2002, 2001, 2008, 2006],
         "amount": [1, 4, 5, 6, 3]}
df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
         "year": [2001, 2002, 2001, 2008, 2007],
         "price": [150, 120, 100, 250, 3000]}
df2 = pd.DataFrame(data2)

# df1, df2의 내용을 확인하세요
print(df1)
print()
print(df2)
print()
# df1 및 df2의 "fruits"를 Key로 내부 결합한 DataFrame을 df3에 대입하세요
df3 = pd.merge(df1, df2, on="fruits", how="inner")

# 출력합니다
# 내부 결합의 동작을 확인합시다
print(df3)


# In[8]:


import numpy as np
import pandas as pd

data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year": [2001, 2002, 2001, 2008, 2006],
         "amount": [1, 4, 5, 6, 3]}

df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
         "year": [2001, 2002, 2001, 2008, 2007],
         "price": [150, 120, 100, 250, 3000]}
df2 = pd.DataFrame(data2)

# df1, df2의 내용을 확인하세요
print(df1)
print()
print(df2)
print()

# df1 및 df2을 "fruits"를 Key로 외부 결합한 DataFrame을 df3에 대입하세요
df3 = pd.merge(df1, df2, on="fruits", how="outer")

# 출력합니다
# 외부 결합의 동작을 확인합시다
print(df3)


# In[9]:


import pandas as pd

# 주문 정보입니다
order_df = pd.DataFrame([[1000, 2546, 103],
                         [1001, 4352, 101],
                         [1002, 342, 101]],
                        columns=["id", "item_id", "customer_id"])

# 고객 정보입니다
customer_df = pd.DataFrame([[101, "Tanaka"],
                            [102, "Suzuki"],
                            [103, "Kato"]],
                           columns=["id", "name"])

# order_df를 바탕으로 "id"를 customer_df와 결합하여 order_df에 대입하세요
order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_on="id", how="inner")

print(order_df)


# In[10]:


import pandas as pd

# 주문 정보입니다
order_df = pd.DataFrame([[1000, 2546, 103],
                         [1001, 4352, 101],
                         [1002, 342, 101]],
                        columns=["id", "item_id", "customer_id"])

# 고객 정보입니다
customer_df = pd.DataFrame([["Tanaka"],
                            ["Suzuki"],
                            ["Kato"]],
                           columns=["name"])
customer_df.index = [101, 102, 103]

# customer_df를 바탕으로 "name"을 order_df와 결합하여 order_df에 대입하세요
order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_index=True, how="inner")

print(order_df)


# In[11]:


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# df의 첫 3행을 취득해 df_head에 대입하세요
df_head = df.head(3)

# df의 끝 3행을 취득해 df_tail에 대입하세요
df_tail = df.tail(3)

# 출력합니다
print(df_head)
print(df_tail)


# In[12]:


import numpy as np
import pandas as pd
import math
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# df의 각 요소를 두 배로 만들어 double_df에 대입하세요
double_df = df * 2 # double_df = df + df도 OK입니다

# df의 각 요소를 제곱하여 square_df에 대입하세요
square_df = df * df #square_df = df**2도 OK입니다

# df의 각 요소의 제곱근을 계산하여 sqrt_df에 대입하세요
sqrt_df = np.sqrt(df) 

# 출력합니다
print(double_df)
print(square_df)
print(sqrt_df)


# In[13]:


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# df의 통계 정보 중 "mean", "max", "min"을 꺼내 df_des에 대입하세요
df_des = df.describe().loc[["mean", "max", "min"]]

print(df_des)


# In[14]:


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# df의 각 행에 대해, 2행 뒤와의 차이를 계산한 DataFrame을 df_diff에 대입하세요
df_diff = df.diff(-2, axis=0)

# df와 df_diff의 내용을 비교해 처리 결과를 확인하세요
print(df)
print(df_diff)


# In[15]:


import pandas as pd

# 도시 정보를 가진 DataFrame을 만듭니다
prefecture_df = pd.DataFrame([["Tokyo", 2190, 13636, "Kanto"], 
                              ["Kanagawa", 2415, 9145, "Kanto"],
                              ["Osaka", 1904, 8837, "Kinki"],
                              ["Kyoto", 4610, 2605, "Kinki"],
                              ["Aichi", 5172, 7505, "Chubu"]],
                             columns=["Prefecture", "Area",
                                      "Population", "Region"])

# 출력합니다
print(prefecture_df)

# prefecture_df을 지역(Region)으로 그룹화하여 grouped_region에 대입하세요
grouped_region = prefecture_df.groupby("Region")

# prefecture_df의 지역별 면적(Area)과 인구(Population)의 평균을 mean_df에 대입하세요
mean_df = grouped_region.mean()

# 출력합니다
print(mean_df)


# In[16]:


import pandas as pd
# 두 DataFrame을 정의합니다
df1 = pd.DataFrame([["apple", "Fruit", 120],
                    ["orange", "Fruit", 60],
                    ["banana", "Fruit", 100],
                    ["pumpkin", "Vegetable", 150],
                    ["potato", "Vegetable", 80]],
                    columns=["Name", "Type", "Price"])

df2 = pd.DataFrame([["onion", "Vegetable", 60],
                    ["carrot", "Vegetable", 50],
                    ["beans", "Vegetable", 100],
                    ["grape", "Fruit", 160],
                    ["kiwifruit", "Fruit", 80]],
                   columns=["Name", "Type", "Price"])

# 여기에 해답을 기술하세요
# 결합합니다
df3 = pd.concat([df1, df2], axis=0)

# 과일만 추출하여 Price로 정렬합니다
df_fruit = df3.loc[df3["Type"] == "Fruit"]
df_fruit = df_fruit.sort_values(by="Price")

# 야채만 추출하여 Price로 정렬합니다
df_veg = df3.loc[df3["Type"] == "Vegetable"]
df_veg = df_veg.sort_values(by="Price")

# 과일과 야채의 상위 세 요소의 Price 합을 각각 계산합니다
print(sum(df_fruit[:3]["Price"]) + sum(df_veg[:3]["Price"]))


# In[17]:


import pandas as pd

index = ["taro", "mike", "kana", "jun", "sachi"]
columns = ["국어", "수학", "사회", "과학", "영어"]
data = [[30, 45, 12, 45, 87], [65, 47, 83, 17, 58], [64, 63, 86, 57, 46,], [38, 47, 62, 91, 63], [65, 36, 85, 94, 36]]
df = pd.DataFrame(data, index=index, columns=columns)

# df에 "체육"이라는 새 열을 만들어 pe_column의 데이터를 추가하세요
pe_column = pd.Series([56, 43, 73, 82, 62], index=["taro", "mike", "kana", "jun", "sachi"])
df["체육"] = pe_column
print(df)
print()

# 수학을 오름차순으로 정렬합니다
df1 = df.sort_values(by="수학", ascending=True)
print(df1)
print()

# df1의 각 요소에 5점을 더하세요
df2 = df1 + 5
print(df2)
print()

# df의 통계 정보 중에서 "mean", "max", "min"을 출력하세요
print(df2.describe().loc[["mean", "max", "min"]])


# In[ ]:




