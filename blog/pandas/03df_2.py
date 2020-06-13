# #pandas 3장

# import pandas as pd
# import numpy as np

# #concat

# def make_random_df(index, columns, seed):
#     np.random.seed(seed)
#     df = pd.DataFrame()
#     for column in columns:
#         df[column] = np.random.choice(range(1,101),len(index))
#     df.index = index
#     return df
# #데이터의 index 길이와, columns 값을 list로 받아보고, random 값을 받는 거기에,
# #seed값을 설정된 함수

# columns = ["apple", "orange", "banana"]
# data1 = make_random_df(range(1,5),columns,0)
# data2 = make_random_df(range(1,5),columns,1)

# df1 = pd.concat([data1,data2],axis=0)
# df2 = pd.concat([data1,data2],axis=1)

# print(data1)
# #    apple  orange  banana
# # 1     45      68      37
# # 2     48      10      88
# # 3     65      84      71
# # 4     68      22      89

# print(data2)
# #    apple  orange  banana
# # 1     38      76      17
# # 2     13       6       2
# # 3     73      80      77
# # 4     10      65      72
# print(df1)
# #    apple  orange  banana
# # 1     45      68      37
# # 2     48      10      88
# # 3     65      84      71
# # 4     68      22      89
# # 1     38      76      17
# # 2     13       6       2
# # 3     73      80      77
# # 4     10      65      72

# print(df2)
# #    apple  orange  banana  apple  orange  banana
# # 1     45      68      37     38      76      17
# # 2     48      10      88     13       6       2
# # 3     65      84      71     73      80      77
# # 4     68      22      89     10      65      72


# #pandas 3장

# import pandas as pd
# import numpy as np

# #concat

# def make_random_df(index, columns, seed):
#     np.random.seed(seed)
#     df = pd.DataFrame()
#     for column in columns:
#         df[column] = np.random.choice(range(1,101),len(index))
#     df.index = index
#     return df
# #데이터의 index 길이와, columns 값을 list로 받아보고, random 값을 받는 거기에,
# #seed값을 설정된 함수

# columns = ["apple", "orange", "banana"]
# data1 = make_random_df(range(1,5),columns,0)
# data2 = make_random_df(range(1,5),columns,1)

# df1 = pd.concat([data1,data2],axis=0)
# df2 = pd.concat([data1,data2],axis=1)

# print(data1)
# #    apple  orange  banana
# # 1     45      68      37
# # 2     48      10      88
# # 3     65      84      71
# # 4     68      22      89

# print(data2)
# #    apple  orange  banana
# # 1     38      76      17
# # 2     13       6       2
# # 3     73      80      77
# # 4     10      65      72
# print(df1)
# #    apple  orange  banana
# # 1     45      68      37
# # 2     48      10      88
# # 3     65      84      71
# # 4     68      22      89
# # 1     38      76      17
# # 2     13       6       2
# # 3     73      80      77
# # 4     10      65      72

# print(df2)
# #    apple  orange  banana  apple  orange  banana
# # 1     45      68      37     38      76      17
# # 2     48      10      88     13       6       2
# # 3     65      84      71     73      80      77
# # 4     68      22      89     10      65      72

# df = pd.concat([data1, data2], axis=1, keys=["x", "y"])
# print(df)

# #              x                   y
# #   apple orange banana apple orange banana
# # 1    45     68     37    38     76     17
# # 2    48     10     88    13      6      2
# # 3    65     84     71    73     80     77
# # 4    68     22     89    10     65     72

# print(df["x"])
# #    apple  orange  banana
# # 1     45      68      37
# # 2     48      10      88
# # 3     65      84      71
# # 4     68      22      89

# print(df["x","apple"])
# # 1    45
# # 2    48
# # 3    65
# # 4    68
# # Name: (x, apple), dtype: int32

# # print(df["apple"])
# # #error
# # #기존에 key값을 넣어서,검색하려고 해도 index(level_index, key) 순서대로 올라가기 때문에 
# # #keys에 들어간 값을 넣어줘야 합니다.

# import numpy as np
# import pandas as pd

# data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
#          "year": [2001, 2002, 2001, 2008, 2006],
#          "amount": [1, 4, 5, 6, 3]}
# df1 = pd.DataFrame(data1)

# data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
#          "year": [2001, 2002, 2001, 2008, 2007],
#          "price": [150, 120, 100, 250, 3000]}
# df2 = pd.DataFrame(data2)

# df3 = pd.merge(df1, df2, on="fruits", how="inner")

# print(df3)
# #        fruits  year_x  amount  year_y  price
# # 0       apple    2001       1    2001    150
# # 1      orange    2002       4    2002    120
# # 2      banana    2001       5    2001    100
# # 3  strawberry    2008       6    2008    250


# import numpy as np
# import pandas as pd

# data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
#          "year": [2001, 2002, 2001, 2008, 2006],
#          "amount": [1, 4, 5, 6, 3]}

# df1 = pd.DataFrame(data1)

# data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
#          "year": [2001, 2002, 2001, 2008, 2007],
#          "price": [150, 120, 100, 250, 3000]}
# df2 = pd.DataFrame(data2)

# # df1, df2의 내용을 확인하세요
# print(df1)
# print()
# print(df2)
# print()

# # df1 및 df2을 "fruits"를 Key로 외부 결합한 DataFrame을 df3에 대입하세요
# df3 = pd.merge(df1, df2, on="fruits", how="outer")

# # 출력합니다
# # 외부 결합의 동작을 확인합시다
# print(df3)

# #        fruits  year_x  amount  year_y   price
# # 0       apple  2001.0     1.0  2001.0   150.0
# # 1      orange  2002.0     4.0  2002.0   120.0
# # 2      banana  2001.0     5.0  2001.0   100.0
# # 3  strawberry  2008.0     6.0  2008.0   250.0
# # 4   kiwifruit  2006.0     3.0     NaN     NaN
# # 5       mango     NaN     NaN  2007.0  3000.0

# import pandas as pd

# order_df = pd.DataFrame([[1000, 2546, 103],
#                          [1001, 4352, 101],
#                          [1002, 342, 101]],
#                         columns=["id", "item_id", "customer_id"])


# customer_df = pd.DataFrame([[101, "Tanaka"],
#                             [102, "Suzuki"],
#                             [103, "Kato"]],
#                            columns=["id", "name"])

# order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_on="id", how="inner")

# print(order_df)
# #    id_x  item_id  customer_id  id_y    name
# # 0  1000     2546          103   103    Kato
# # 1  1001     4352          101   101  Tanaka
# # 2  1002      342          101   101  Tanaka


# import pandas as pd

# order_df = pd.DataFrame([[1000, 2546, 103],
#                          [1001, 4352, 101],
#                          [1002, 342, 101]],
#                         columns=["id", "item_id", "customer_id"])

# customer_df = pd.DataFrame([["Tanaka"],
#                             ["Suzuki"],
#                             ["Kato"]],
#                            columns=["name"])
# customer_df.index = [101, 102, 103]

# order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_index=True, how="inner")

# print(order_df)
# #      id  item_id  customer_id    name
# # 0  1000     2546          103    Kato
# # 1  1001     4352          101  Tanaka
# # 2  1002      342          101  Tanaka

# import numpy as np
# import pandas as pd
# np.random.seed(0)
# columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# df = pd.DataFrame()
# for column in columns:
#     df[column] = np.random.choice(range(1, 11), 10)
# df.index = range(1, 11)

# df_head = df.head(3)

# df_tail = df.tail(3)

# print(df_head)
# #    apple  orange  banana  strawberry  kiwifruit
# # 1      6       8       6           3         10
# # 2      1       7      10           4         10
# # 3      4       9       9           9          1

# print(df_tail)
# #     apple  orange  banana  strawberry  kiwifruit
# # 8       6       8       4           8          8
# # 9       3       9       6           1          3
# # 10      5       2       1           2          1

# import numpy as np
# import pandas as pd
# np.random.seed(0)
# columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# df = pd.DataFrame()
# for column in columns:
#     df[column] = np.random.choice(range(1, 11), 10)
# df.index = range(1, 11)

# double_df = df * 2 # double_df = df + df도 OK입니다

# square_df = df * df #square_df = df**2도 OK입니다

# sqrt_df = np.sqrt(df) 

# print(double_df)
# #     apple  orange  banana  strawberry  kiwifruit
# # 1      12      16      12           6         20
# # 2       2      14      20           8         20
# # 3       8      18      18          18          2
# # 4       8      18      20           4         10
# # 5      16       4      10           8         16
# # 6      20      14       8           8          8
# # 7       8      16       2           8          6
# # 8      12      16       8          16         16
# # 9       6      18      12           2          6
# # 10     10       4       2           4          2

# print(square_df)
# #     apple  orange  banana  strawberry  kiwifruit
# # 1      36      64      36           9        100
# # 2       1      49     100          16        100
# # 3      16      81      81          81          1
# # 4      16      81     100           4         25
# # 5      64       4      25          16         64
# # 6     100      49      16          16         16
# # 7      16      64       1          16          9
# # 8      36      64      16          64         64
# # 9       9      81      36           1          9
# # 10     25       4       1           4          1

# print(sqrt_df)
# #        apple    orange    banana  strawberry  kiwifruit
# # 1   2.449490  2.828427  2.449490    1.732051   3.162278
# # 2   1.000000  2.645751  3.162278    2.000000   3.162278
# # 3   2.000000  3.000000  3.000000    3.000000   1.000000
# # 4   2.000000  3.000000  3.162278    1.414214   2.236068
# # 5   2.828427  1.414214  2.236068    2.000000   2.828427
# # 6   3.162278  2.645751  2.000000    2.000000   2.000000
# # 7   2.000000  2.828427  1.000000    2.000000   1.732051
# # 8   2.449490  2.828427  2.000000    2.828427   2.828427
# # 9   1.732051  3.000000  2.449490    1.000000   1.732051
# # 10  2.236068  1.414214  1.000000    1.414214   1.000000


import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10 entries, 1 to 10
# Data columns (total 5 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   apple       10 non-null     int32
#  1   orange      10 non-null     int32
#  2   banana      10 non-null     int32
#  3   strawberry  10 non-null     int32
#  4   kiwifruit   10 non-null     int32
# dtypes: int32(5)
# memory usage: 332.0 bytes

# import numpy as np
# import pandas as pd
# np.random.seed(0)
# columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]




# # DataFrame을 생성하고 열을 추가합니다
# df = pd.DataFrame()
# for column in columns:
#     df[column] = np.random.choice(range(1, 11), 10)
# df.index = range(1, 11)

# # df의 통계 정보 중 "mean", "max", "min"을 꺼내 df_des에 대입하세요
# df_des = df.describe()

# print(df_des)
# #            apple     orange     banana  strawberry  kiwifruit
# # count  10.000000  10.000000  10.000000   10.000000  10.000000
# # mean    5.100000   6.900000   5.600000    4.100000   5.300000
# # std     2.558211   2.685351   3.306559    2.558211   3.465705
# # min     1.000000   2.000000   1.000000    1.000000   1.000000
# # 25%     4.000000   7.000000   4.000000    2.250000   3.000000
# # 50%     4.500000   8.000000   5.500000    4.000000   4.500000
# # 75%     6.000000   8.750000   8.250000    4.000000   8.000000
# # max    10.000000   9.000000  10.000000    9.000000  10.000000

# df_des = df.describe().loc[["mean", "max", "min"]]
# print(df_des)
# #       apple  orange  banana  strawberry  kiwifruit
# # mean    5.1     6.9     5.6         4.1        5.3
# # max    10.0     9.0    10.0         9.0       10.0
# # min     1.0     2.0     1.0         1.0        1.0

# import numpy as np
# import pandas as pd
# np.random.seed(0)
# columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# df = pd.DataFrame()
# for column in columns:
#     df[column] = np.random.choice(range(1, 11), 10)
# df.index = range(1, 11)

# df_diff = df.diff(-2, axis=0)

# print(df)
# #     apple  orange  banana  strawberry  kiwifruit
# # 1       6       8       6           3         10
# # 2       1       7      10           4         10
# # 3       4       9       9           9          1
# # 4       4       9      10           2          5
# # 5       8       2       5           4          8
# # 6      10       7       4           4          4
# # 7       4       8       1           4          3
# # 8       6       8       4           8          8
# # 9       3       9       6           1          3
# # 10      5       2       1           2          1

# print(df_diff)
# #     apple  orange  banana  strawberry  kiwifruit
# # 1     2.0    -1.0    -3.0        -6.0        9.0
# # 2    -3.0    -2.0     0.0         2.0        5.0
# # 3    -4.0     7.0     4.0         5.0       -7.0
# # 4    -6.0     2.0     6.0        -2.0        1.0
# # 5     4.0    -6.0     4.0         0.0        5.0
# # 6     4.0    -1.0     0.0        -4.0       -4.0
# # 7     1.0    -1.0    -5.0         3.0        0.0
# # 8     1.0     6.0     3.0         6.0        7.0
# # 9     NaN     NaN     NaN         NaN        NaN
# # 10    NaN     NaN     NaN         NaN        NaN


# import pandas as pd

# prefecture_df = pd.DataFrame([["Tokyo", 2190, 13636, "Kanto"], 
#                               ["Kanagawa", 2415, 9145, "Kanto"],
#                               ["Osaka", 1904, 8837, "Kinki"],
#                               ["Kyoto", 4610, 2605, "Kinki"],
#                               ["Aichi", 5172, 7505, "Chubu"]],
#                              columns=["Prefecture", "Area",
#                                       "Population", "Region"])

# print(prefecture_df)

# grouped_region = prefecture_df.groupby("Region")

# mean_df = grouped_region.mean()

# # 출력합니다
# print(mean_df)
# #           Area  Population
# # Region
# # Chubu   5172.0      7505.0
# # Kanto   2302.5     11390.5
# # Kinki   3257.0      5721.0

# #
# df = pd.read_csv("winequality-white.csv",index_col=0,header=0,encoding="cp949",sep=";")

# print(df.head())
# #                volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide
# # fixed acidity
# # 7.0                        0.27         0.36            20.7      0.045                 45.0
# # 6.3                        0.30         0.34             1.6      0.049                 14.0
# # 8.1                        0.28         0.40             6.9      0.050                 30.0
# # 7.2                        0.23         0.32             8.5      0.058                 47.0
# # 7.2                        0.23         0.32             8.5      0.058                 47.0

# df2 = pd.read_csv("winequality-white.csv",index_col=0,header=None,names=list(range(1,14)),
#                   skiprows=1, encoding="cp949",sep=";")

# print(df2.head())

# #        2     3     4      5     6      7       8     9     10    11  12  13
# # 1
# # 7.0  0.27  0.36  20.7  0.045  45.0  170.0  1.0010  3.00  0.45   8.8   6 NaN
# # 6.3  0.30  0.34   1.6  0.049  14.0  132.0  0.9940  3.30  0.49   9.5   6 NaN
# # 8.1  0.28  0.40   6.9  0.050  30.0   97.0  0.9951  3.26  0.44  10.1   6 NaN
# # 7.2  0.23  0.32   8.5  0.058  47.0  186.0  0.9956  3.19  0.40   9.9   6 NaN
# # 7.2  0.23  0.32   8.5  0.058  47.0  186.0  0.9956  3.19  0.40   9.9   6 NaN

# df.to_csv("submission.csv",sep=",",index=True, header=True,
#           columns=["volatile acidity"],index_label="index_name")

# df = pd.read_csv("submission.csv",index_col=0,header=0)

# print(df.head())

# #             volatile acidity
# # index_name
# # 7.0                     0.27
# # 6.3                     0.30
# # 8.1                     0.28
# # 7.2                     0.23
# # 7.2                     0.23