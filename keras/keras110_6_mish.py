import numpy as np
import matplotlib.pyplot as plt
from keras.activations import softplus

x = np.arange(-5, 5, 0.1)


def mish(x):
    return x* np.tanh(softplus(x))
y = mish()    


plt.plot(x,y)
plt.grid()
plt.show()