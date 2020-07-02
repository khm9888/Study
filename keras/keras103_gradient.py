import matplotlib.pyplot as plt
import numpy as np

gradient = lambda x :x**2-4*x+6

x= np.linspace(-1,6,100)
y= gradient(x)

plt.plot(x,y,"k-")
plt.plot(2,2,"sk")
plt.grid()


plt.show()