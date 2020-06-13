# import pandas as pd

# df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

# # df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols",
# # "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# print(df)


import pandas as pd

df = pd.read_csv(
"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None)
print(df)
#        0    1    2    3               4
# 0    5.1  3.5  1.4  0.2     Iris-setosa
# 1    4.9  3.0  1.4  0.2     Iris-setosa
# 2    4.7  3.2  1.3  0.2     Iris-setosa
# 3    4.6  3.1  1.5  0.2     Iris-setosa
# 4    5.0  3.6  1.4  0.2     Iris-setosa
# ..   ...  ...  ...  ...             ...
# 145  6.7  3.0  5.2  2.3  Iris-virginica
# 146  6.3  2.5  5.0  1.9  Iris-virginica
# 147  6.5  3.0  5.2  2.0  Iris-virginica
# 148  6.2  3.4  5.4  2.3  Iris-virginica
# 149  5.9  3.0  5.1  1.8  Iris-virginica

df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

print(df)
#      sepal length  sepal width  petal length  petal width           class
# 0             5.1          3.5           1.4          0.2     Iris-setosa
# 1             4.9          3.0           1.4          0.2     Iris-setosa
# 2             4.7          3.2           1.3          0.2     Iris-setosa
# 3             4.6          3.1           1.5          0.2     Iris-setosa
# 4             5.0          3.6           1.4          0.2     Iris-setosa
# ..            ...          ...           ...          ...             ...
# 145           6.7          3.0           5.2          2.3  Iris-virginica
# 146           6.3          2.5           5.0          1.9  Iris-virginica
# 147           6.5          3.0           5.2          2.0  Iris-virginica
# 148           6.2          3.4           5.4          2.3  Iris-virginica
# 149           5.9          3.0           5.1          1.8  Iris-virginica

#파일쓰기
import csv

with open("csv0.csv", "w") as csvfile:
    writer = csv.writer(csvfile, lineterminator="\n")

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

# city,year,season
# Nagano,1998,winter
# Sydney,2000,summer
# Salt Lake City,2002,winter
# Athens,2004,summer
# Torino,2006,winter
# Beijing,2008,summer
# Vancouver,2010,winter
# London,2012,summer
# Sochi,2014,winter
# Rio de Janeiro,2016,summer

import pandas as pd

data = {"city": ["Nagano", "Sydney", "Salt Lake City", "Athens", "Torino", "Beijing", "Vancouver", "London", "Sochi", "Rio de Janeiro"],
        "year": [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016],
        "season": ["winter", "summer", "winter", "summer", "winter", "summer", "winter", "summer", "winter", "summer"]}

df = pd.DataFrame(data)

df.to_csv("csv1.csv")

# ,city,year,season
# 0,Nagano,1998,winter
# 1,Sydney,2000,summer
# 2,Salt Lake City,2002,winter
# 3,Athens,2004,summer
# 4,Torino,2006,winter
# 5,Beijing,2008,summer
# 6,Vancouver,2010,winter
# 7,London,2012,summer
# 8,Sochi,2014,winter
# 9,Rio de Janeiro,2016,summer

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

attri_data_frame1 = attri_data_frame1.append(attri_data_frame2)

print(attri_data_frame1)

#     ID      city  birth_year     name
# 0  100     Tokyo        1990  Hiroshi
# 1  101     Osaka        1989    Akiko
# 2  102     Kyoto        1992     Yuki
# 3  103  Hokkaido        1997   Satoru
# 4  104     Tokyo        1982   Steeve
# 5  106     Tokyo        1991   Mituru
# 6  108     Osaka        1988      Aoi
# 7  110     Kyoto        1990    Tarou
# 8  111  Hokkaido        1995   Suguru
# 9  113     Tokyo        1981   Mitsuo
# 0  107    Sendai        1994      NaN
# 1  109    Nagoya        1988      NaN

attri_data_frame1=attri_data_frame1.sort_values("ID").reset_index(drop=True)

print(attri_data_frame1)
#     ID      city  birth_year     name
# 0  100     Tokyo        1990  Hiroshi
# 1  101     Osaka        1989    Akiko
# 2  102     Kyoto        1992     Yuki
# 3  103  Hokkaido        1997   Satoru
# 4  104     Tokyo        1982   Steeve
# 5  106     Tokyo        1991   Mituru
# 0  107    Sendai        1994      NaN
# 6  108     Osaka        1988      Aoi
# 1  109    Nagoya        1988      NaN
# 7  110     Kyoto        1990    Tarou
# 8  111  Hokkaido        1995   Suguru
# 9  113     Tokyo        1981   Mitsuo

import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA 

print(sample_data_frame)

#           0         1         2         3
# 0  0.929104  0.470784  0.124563  0.794093
# 1       NaN  0.666536  0.582206  0.660878
# 2  0.421970  0.710467       NaN  0.647801
# 3  0.716990  0.651654  0.419760  0.442970
# 4  0.292251  0.280757  0.126363  0.018312
# 5  0.284272  0.393644  0.660524       NaN
# 6  0.065219  0.772722  0.536255       NaN
# 7  0.664015  0.567853  0.491800       NaN
# 8  0.085432  0.723125  0.719816       NaN
# 9  0.671783  0.455025  0.276573       NaN

sample_data_frame=sample_data_frame.dropna()

print(sample_data_frame)
#           0         1         2         3
# 0  0.098183  0.469856  0.267726  0.884424
# 3  0.410031  0.874025  0.500631  0.888595
# 4  0.222625  0.452477  0.013709  0.637019

import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA 

print(sample_data_frame)

sample_data_frame=sample_data_frame[[0,1,2]].dropna()

print(sample_data_frame)
#           0         1         2
# 0  0.665529  0.559377  0.468249
# 3  0.949344  0.645021  0.612441
# 4  0.861691  0.353130  0.154239
# 5  0.781304  0.185882  0.574889
# 6  0.034871  0.352714  0.711653
# 7  0.077110  0.484130  0.400722
# 8  0.940951  0.839424  0.030412
# 9  0.122323  0.524014  0.546039


import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA

print(sample_data_frame)
#           0         1         2         3
# 0  0.379355  0.012638  0.135102  0.122207
# 1       NaN  0.855007  0.599022  0.884963
# 2  0.418225  0.422230       NaN  0.673295
# 3  0.059809  0.682895  0.596560  0.004831
# 4  0.645347  0.756897  0.848214  0.498483
# 5  0.073658  0.195947  0.524592       NaN
# 6  0.817547  0.752998  0.888289       NaN
# 7  0.230461  0.258391  0.831407       NaN
# 8  0.406798  0.701894  0.490686       NaN
# 9  0.519711  0.215536  0.027628       NaN

sample_data_frame=sample_data_frame.fillna(0)

print(sample_data_frame)
#           0         1         2         3
# 0  0.379355  0.012638  0.135102  0.122207
# 1  0.000000  0.855007  0.599022  0.884963
# 2  0.418225  0.422230  0.000000  0.673295
# 3  0.059809  0.682895  0.596560  0.004831
# 4  0.645347  0.756897  0.848214  0.498483
# 5  0.073658  0.195947  0.524592  0.000000
# 6  0.817547  0.752998  0.888289  0.000000
# 7  0.230461  0.258391  0.831407  0.000000
# 8  0.406798  0.701894  0.490686  0.000000
# 9  0.519711  0.215536  0.027628  0.000000

# sample_data_frame=sample_data_frame.fillna(sample_data_frame.mean())

# print(sample_data_frame)


# #           0         1         2         3
# # 0  0.888733  0.629492  0.296944  0.745948
# # 1  0.525436  0.073730  0.044046  0.093941
# # 2  0.492419  0.849434  0.322719  0.517244
# # 3  0.205752  0.075842  0.014893  0.795734
# # 4  0.483072  0.211664  0.314218  0.556719
# # 5  0.998169  0.054442  0.027553  0.541917
# # 6  0.754403  0.086043  0.891892  0.541917
# # 7  0.271241  0.391135  0.523385  0.541917
# # 8  0.118624  0.600650  0.388737  0.541917
# # 9  0.516510  0.526758  0.402807  0.541917

# sample_data_frame=sample_data_frame.fillna(method='ffill' )
# print(sample_data_frame)
# #           0         1         2         3
# # 0  0.153887  0.363813  0.184375  0.888240
# # 1  0.153887  0.236442  0.491826  0.298659
# # 2  0.264396  0.504744  0.491826  0.457869
# # 3  0.144748  0.250124  0.049832  0.018724
# # 4  0.178839  0.996768  0.300089  0.999276
# # 5  0.135913  0.598578  0.446690  0.999276
# # 6  0.501318  0.244950  0.294699  0.999276
# # 7  0.021053  0.177305  0.795804  0.999276
# # 8  0.355963  0.074339  0.774032  0.999276
# # 9  0.275384  0.930521  0.514588  0.999276


# sample_data_frame=sample_data_frame.fillna(method='bfill')

# print(sample_data_frame)

# #           0         1         2         3
# # 0  0.566666  0.465278  0.780942  0.391425
# # 1  0.545889  0.902538  0.405609  0.540805
# # 2  0.545889  0.125099  0.846894  0.083072
# # 3  0.351282  0.200958  0.846894  0.058633
# # 4  0.800475  0.901855  0.601364  0.271015
# # 5  0.219818  0.311200  0.804521       NaN
# # 6  0.274299  0.758538  0.143681       NaN
# # 7  0.295133  0.095120  0.931013       NaN
# # 8  0.182091  0.416574  0.952512       NaN
# # 9  0.413194  0.250391  0.890873       NaN

print(sample_data_frame.std())


import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6], 
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b"]}) 

print(dupli_data)
#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 5     4    c
# 6     6    b
# 7     6    b


print(dupli_data.duplicated())
# 0    False
# 1    False
# 2    False
# 3    False
# 4    False
# 5     True
# 6    False
# 7     True
# dtype: bool

dupli_data = dupli_data.drop_duplicates()

print(dupli_data)
#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 6     6    b