import numpy as np
from keras.activations import softplus

def mish(x):
    return x* np.tanh(softplus(x))
   
mish(model)