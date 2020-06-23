import numpy as np

x=np.random.randint(1,10000,100)
x=x.reshape(-1,10)

# [[1,2,3,4],[4,5,6,7]]


# x = [1,2,3,4,10000,6,7,5000,90,100]
# x=[1,20000,3,4,5,6,7,8,9000,100]
# print(x.shape)

def outliners(data):
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    upper_bound = q3+iqr*1.5
    lower_bound = q1-iqr*1.5
    return np.where((data>upper_bound) | (data<lower_bound))

# print((x[:,0]))
# print(outliners(x[:,0]))


print(outliners(x))
