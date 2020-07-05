import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


# x = np.arange(-5,5,0.1)
# y = relu(x)

# print(x.shape,y.shape)

# plt.plot(x,y)
# plt.grid()
# plt.show()

x = np.arange(1,5,1)
y = softmax(x)

ratio = y

plt.pie(ratio,labels=ratio,shadow=True,startangle=90)
plt.show()

# x = np.arange(-5,5,0.1)
# y = np.tan(-x)

# print(x.shape,y.shape)

# plt.plot(x,y)
# plt.grid()
# plt.show()