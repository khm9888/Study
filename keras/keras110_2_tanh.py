import numpy as np
import matplotlib.pyplot as plt

# def tanh(x):
#     return 1/(1+np.exp(-x))


x = np.arange(-5,5,0.1)
y = np.tanh(-x)

print(x.shape,y.shape)

plt.plot(x,y)
plt.grid()
plt.show()

# x = np.arange(-5,5,0.1)
# y = np.tan(-x)

# print(x.shape,y.shape)

# plt.plot(x,y)
# plt.grid()
# plt.show()