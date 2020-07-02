import numpy as np
import matplotlib.pyplot as plt

def selu(l,alpha=2):
    values=[]
    for x in l:
        if x<0: 
            value=alpha*(np.exp(x)-1)
        else: 
            value=x
        values.append(value)
    return np.array(values)



# x = np.arange(-5,5,0.1)
# y = relu(x)

# print(x.shape,y.shape)

# plt.plot(x,y)
# plt.grid()
# plt.show()

x = np.arange(-5,5,0.1)
y = selu(x)

# print(x.shape,y.shape)

plt.plot(x,y)
plt.grid()
plt.show()

# x = np.arange(-5,5,0.1)
# y = np.tan(-x)

# print(x.shape,y.shape)

# plt.plot(x,y)
# plt.grid()
# plt.show()