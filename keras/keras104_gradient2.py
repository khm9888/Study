import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100)
gradient = lambda x : 2*x - 4
y = f(x)
print(y)



# plot
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
# plt.show()


x0 = 0.0
MaxIter = 10
learning_rate = 0.25

print("step\tx\tf(x)")
print(f"{0:02d}\t{x0:6.5f}\t{f(x0):6.5f}")

# 경사하강법
for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1
    print(f"{(i+1):02d}\t{x0:6.5f}\t{f(x0):6.5f}")