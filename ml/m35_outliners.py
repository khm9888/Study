
import numpy as np

def outliner(data):
    quantile_1,quantile_3 = np.percentile(data,[25,75])
    print("quantile_1")
    print(quantile_1)
    print("quantile_3")
    print(quantile_3)
    iqr = quantile_3-quantile_1
    lower_bound = quantile_1 - (iqr *1.5) 
    upper_bound = quantile_3 + (iqr *1.5)
    return np.where((data>upper_bound) | (data<lower_bound))

a = np.array([1,2,3,4,10000,6,7,6000,90,100])
b=outliner(a) 

print(f"이상치의 위치 : {b}")

#실습, 행렬을 입력해서, 컬럼별로 이상치 발견하는 함수를 구현하시오