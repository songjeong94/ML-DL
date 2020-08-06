
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(1, 1, 1)

major_ticks = np.arange(0, 29, 5)
minor_ticks = np.arange(0, 29, 1)

ax.set_xticks(major_ticks)
ax.set_xticks(major_ticks, minor = True)
ax.set_yticks(major_ticks)
ax.set_yticks(major_ticks, minor = True)

ax.grid(which='both')

ax.grid(which = 'minor', alpha = 0.2)
ax.grid(which = 'major', alpha = 0.5)

ax.imshow(x_test[1], cmap=plt.cm.binary)

plt.show()

print(y_test[1]) 
print(x_test[1]) 
