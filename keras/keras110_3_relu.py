import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)


def leaky_relu(x):
    return np.maximum(0.1*x,x)




x = np.arange(-5,5,0.1)
y = relu(x)

print(x.shape,y.shape)

plt.plot(x,y)
plt.grid()
plt.show()

# x = np.arange(-5,5,0.1)
# y = leaky_relu(x)

# print(x.shape,y.shape)
# plt.plot(x,y)
# plt.grid()
# plt.show()

# x = np.arange(-5,5,0.1)
# y = np.tan(-x)

# print(x.shape,y.shape)

# plt.plot(x,y)
# plt.grid()
# plt.show()