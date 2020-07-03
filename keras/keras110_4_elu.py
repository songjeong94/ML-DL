import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def elu_func(x): # ELU(Exponential linear unit)
    return (x>=0)*x + (x<0)*0.01*(np.exp(x)-1)

y = elu_func(x)

plt.plot(x,y)
plt.grid()
plt.show()
