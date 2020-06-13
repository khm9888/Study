# import numpy as np

# scores = [[10,20,30],[20,30,40,50]]

# # print(scores[1])#20

# subject_scores = {"math": 50, "korean":40}

# # print(subject_scores["korean"])#40

# import pandas as pd
# #pd라고 줄여서 쓰는 편입니다.

# # scores=pd.Series(scores)

# # print(scores)

# # print(scores[2])

# # print(f"scores.index:{list(scores.index)}")
# # print(f"scores.keys:{list(scores.keys())}")
# # # print(f"scores.keys:{scores.keys()}")

# # scores=pd.Series(scores,["zero","one"])

# # print(scores[1])#[20, 30, 40]
# # print(scores["one"])#[20, 30, 40]

# # print(scores[1][1])#30

# # scores = scores.append(pd.Series([40],index=["zero"]))

# # scores=scores.drop("zero")

# # print(scores)

# # zero    [10, 20, 30]
# # one     [20, 30, 40]
# # dtype: object

# # zero    [10, 20, 30]
# # one     [20, 30, 40]
# # dtype: object

# # zero    [10, 20, 30]
# # one     [20, 30, 40]
# # dtype: object

# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")
# # print(f"scores.index:{li)}")

# # # zero    10
# # # one     20
# # # two     30
# # # dtype: int64
# # print(f"scores.keys:{l())}")

# # subject_scores = pd.Series(subject_scores)

# # print(subject_scores)


# # print(subject_scores[0])#50
# # print(subject_scores["korean"])#40


# # item= scores.sort_index()

# # print(item)


# # df_1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
# # 'B': ['B0', 'B1', 'B2'],
# # 'C': ['C0', 'C1', 'C2'],
# # 'D': ['D0', 'D1', 'D2']},
# # index=[0, 1, 2])

# # df_2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
# # 'B': ['B3', 'B4', 'B5'],
# # 'C': ['C3', 'C4', 'C5'],
# # 'D': ['D3', 'D4', 'D5']})

# # print(df_1)
# # print(df_2)

# # item = df_1.append(df_2)

# # print(item)

# #     A   B   C   D
# # 0  A0  B0  C0  D0
# # 1  A1  B1  C1  D1
# # 2  A2  B2  C2  D2
# #     A   B   C   D
# # 3  A3  B3  C3  D3
# # 4  A4  B4  C4  D4
# # 5  A5  B5  C5  D5
# #     A   B   C   D
# # 0  A0  B0  C0  D0
# # 1  A1  B1  C1  D1
# # 2  A2  B2  C2  D2
# # 3  A3  B3  C3  D3
# # 4  A4  B4  C4  D4
# # 5  A5  B5  C5  D5

# import pandas as pd

# data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
#         "year": [2001, 2002, 2001, 2008, 2006],
#         "time": [1, 4, 5, 6, 3]}
# df = pd.DataFrame(data,index ="a b c d e".split())


# # df = df.loc["b":"c"]["time"]#column과 같은 경우
# df = df.loc[["b","c"],["time","year"]]#column과 같은 경우
# # df = df.iloc[[1:],[1:]]#column과 같은 경우
# print(df)
# # df = df.iloc[1:][:]#column과 같은 경우
# # print(type(df))
# # print(df)



# # df = df.iloc[1:][:]#column과 같은 경우
# # print(type(df))
# # print(df)
# # # time  year
# # 1     4  2002
# # 2     5  2001

import pandas as pd

data = [10, 5, 8, 12, 3]
series = pd.Series(data)

print(series[series.index%2==0])

# 0    10
# 2     8
# 4     3
# dtype: int64

print(series[series.values%2==0])

# 0    10
# 2     8
# 3    12
# dtype: int64

conditions = [True, True, False, False, False]
print(series[conditions])
# 0    10
# 1     5
# dtype: int64

