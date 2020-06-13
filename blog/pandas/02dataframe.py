# import pandas as pd

# data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
#         "year": [2001, 2002, 2001, 2008, 2006],
#         "time": [1, 4, 5, 6, 3]}
# df = pd.DataFrame(data)
# series = pd.Series(["mango", 2008, 7], index=["fruits", "year", "time"])


# print("series")
# print(series)

# # series
# # fruits    mango
# # year       2008
# # time          7
# # dtype: object

# df = df.append(series, ignore_index=True)
# #ignore_index 매개변수를 True로 설정해야지 index를 무시하고 새로 추가가 가능합니다.

# print(df)

# # fruits  year  time
# # 0       apple  2001     1
# # 1      orange  2002     4
# # 2      banana  2001     5
# # 3  strawberry  2008     6
# # 4   kiwifruit  2006     3
# # 5       mango  2008     7

# import pandas as pd

# # Series 데이터입니다
# fruits = {"orange": 2,"banana": 3}
# print(pd.Series(fruits))


# In[3]:

# import pandas as pd

# data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
#         "year": [2001, 2002, 2001, 2008, 2006],
#         "time": [1, 4, 5, 6, 3]}

# df = pd.DataFrame(data)
# print(df)
# #        fruits  year  time
# # 0       apple  2001     1
# # 1      orange  2002     4
# # 2      banana  2001     5
# # 3  strawberry  2008     6
# # 4   kiwifruit  2006     3

# # #1번째
# # df = df.iloc[[1, 3], [0, 2]]

# # print(df)
# # #        fruits  time
# # # 1      orange     4
# # # 3  strawberry     6

# # print(type(df))
# # #<class 'pandas.core.frame.DataFrame'>

# # #2번째 경우
# # df = df.iloc[[1,3]]["year","time"]
# # print(df)
# # # 1    2002
# # # 3    2008
# # print(type(df))
# # # Name: year, dtype: int64
# # # <class 'pandas.core.series.Series'>

# # #error
# # df = df.iloc[[1, 3]][0]
# # print(df)


# df = df["year"]
# print(type(df))
# # <class 'pandas.core.series.Series'>
# print(df)
# # 0    2001
# # 1    2002
# # 2    2001
# # 3    2008
# # 4    2006
# print(df[2])
# # 2001


#행/열 삭제

# import pandas as pd

# data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
#         "year": [2001, 2002, 2001, 2008, 2006],
#         "time": [1, 4, 5, 6, 3]}

# df = pd.DataFrame(data)

# # #1번째
# # df = df.drop(1)

# # print(df)
# # #        fruits  year  time
# # # 0       apple  2001     1
# # # 2      banana  2001     5
# # # 3  strawberry  2008     6
# # # 4   kiwifruit  2006     3

# #2번째
# df = df.drop("year",axis=1)

# print(df)
# #        fruits  time
# # 0       apple     1
# # 1      orange     4
# # 2      banana     5
# # 3  strawberry     6
# # 4   kiwifruit     3

#행/열 정렬

import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print(df)
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3

df = df.sort_values("fruits")
print(df)
#        fruits  year  time
# 0       apple  2001     1
# 2      banana  2001     5
# 4   kiwifruit  2006     3
# 1      orange  2002     4
# 3  strawberry  2008     6
df = df.sort_values(1,axis=1)
print(df)


df = df.sort_index(axis=1)
print(df)
#        fruits  time  year
# 0       apple     1  2001
# 2      banana     5  2001
# 4   kiwifruit     3  2006
# 1      orange     4  2002
# 3  strawberry     6  2008

df = df.sort_index()
print(df)

#        fruits  time  year
# 0       apple     1  2001
# 1      orange     4  2002
# 2      banana     5  2001
# 3  strawberry     6  2008
# 4   kiwifruit     3  2006

import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3

print(df.index % 2 == 0)
print()
print(df[df.index % 2 == 0])

# [ True False  True False  True]

#       fruits  year  time
# 0      apple  2001     1
# 2     banana  2001     5
# 4  kiwifruit  2006     3