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
               "city": ["서울", "대전", "광주", "부산", "서울", "서울", "대전", "광주", "부산", "서울"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["영이","순돌","짱구","태양","션","유리","현아","태식","민수","호식"] }

attri_data_frame1 = DataFrame(attri_data1)

attri_data2 = {"ID": ["107", "109"],
               "city": ["봉화", "전주"],
               "birth_year": [1994, 1988]}

attri_data_frame2 = DataFrame(attri_data2)

attri_data_frame1 = attri_data_frame1.append(attri_data_frame2)

print(attri_data_frame1)


#    ID city  birth_year name
# 0  100   서울        1990   영이
# 1  101   대전        1989   순돌
# 2  102   광주        1992   짱구
# 3  103   부산        1997   태양
# 4  104   서울        1982    션
# 5  106   서울        1991   유리
# 6  108   대전        1988   현아
# 7  110   광주        1990   태식
# 8  111   부산        1995   민수
# 9  113   서울        1981   호식
# 0  107   봉화        1994  NaN
# 1  109   전주        1988  NaN

attri_data_frame1=attri_data_frame1.sort_values("ID").reset_index(drop=True)

print(attri_data_frame1)
#      ID city  birth_year name
# 0   100   서울        1990   영이
# 1   101   대전        1989   순돌
# 2   102   광주        1992   짱구
# 3   103   부산        1997   태양
# 4   104   서울        1982    션
# 5   106   서울        1991   유리
# 6   107   봉화        1994  NaN
# 7   108   대전        1988   현아
# 8   109   전주        1988  NaN
# 9   110   광주        1990   태식
# 10  111   부산        1995   민수
# 11  113   서울        1981   호식

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


city_map ={"서울":"서울", 
           "부산":"경상도", 
           "대전":"충청도", 
           "광주":"전라도",
           "전주":"전라도",
           "봉화":"경상도",}
print(city_map)
# {'서울': '서울', '부산': '경상도', '대전': '충청도', '광주': '전라도'}


attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "대전", "광주", "부산", "서울", "서울", "대전", "광주", "부산", "서울"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["영이","순돌","짱구","태양","션","유리","현아","태식","민수","호식"] }
attri_data_frame1 = DataFrame(attri_data1)

attri_data2 = {"ID": ["107", "109"],
               "city": ["봉화", "전주"],
               "birth_year": [1994, 1988]}

attri_data_frame2 = DataFrame(attri_data2)

attri_data_frame1 = attri_data_frame1.append(attri_data_frame2)

attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)
print(attri_data_frame1)

#     ID city  birth_year name region
# 0  100   서울        1990   영이     서울
# 1  101   대전        1989   순돌    충청도
# 2  102   광주        1992   짱구    전라도
# 3  103   부산        1997   태양    경상도
# 4  104   서울        1982    션     서울
# 5  106   서울        1991   유리     서울
# 6  108   대전        1988   현아    충청도
# 7  110   광주        1990   태식    전라도
# 8  111   부산        1995   민수    경상도
# 9  113   서울        1981   호식     서울
# 0  107   봉화        1994  NaN    경상도
# 1  109   전주        1988  NaN    전라도

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "대전", "광주", "부산", "서울", "서울", "대전", "광주", "부산", "서울"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["영이","순돌","짱구","태양","션","유리","현아","태식","민수","호식"] }
attri_data_frame1 = DataFrame(attri_data1)


birth_year_bins = [1980, 1985, 1990, 1995, 2000]

birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)
print(birth_year_cut_data)

# 0    (1985, 1990]
# 1    (1985, 1990]
# 2    (1990, 1995]
# 3    (1995, 2000]
# 4    (1980, 1985]
# 5    (1990, 1995]
# 6    (1985, 1990]
# 7    (1985, 1990]
# 8    (1990, 1995]
# 9    (1980, 1985]
# Name: birth_year, dtype: category
# Categories (4, interval[int64]): [(1980, 1985] < (1985, 1990] < (1990, 1995] < (1995, 2000]]

print(pd.value_counts(birth_year_cut_data))
# (1985, 1990]    4
# (1990, 1995]    3
# (1980, 1985]    2
# (1995, 2000]    1
# Name: birth_year, dtype: int64

# group_names = ["first1980", "second1980", "first1990", "second1990"]
# birth_year_cut_data = pd.cut(attri_data_frame1.birth_year,birth_year_bins,labels = group_names)
# print(birth_year_cut_data)
# # 0    second1980
# # 1    second1980
# # 2     first1990
# # 3    second1990
# # 4     first1980
# # 5     first1990
# # 6    second1980
# # 7    second1980
# # 8     first1990
# # 9     first1980
# # Name: birth_year, dtype: category
# print(pd.value_counts(birth_year_cut_data))
# # second1980    4
# # first1990     3
# # first1980     2
# # second1990    1
# # Name: birth_year, dtype: int64

birth_year_cut_data=pd.cut(attri_data_frame1.birth_year, 2)
print(pd.value_counts(birth_year_cut_data))
# (1989.0, 1997.0]      6
# (1980.984, 1989.0]    4
# Name: birth_year, dtype: int64