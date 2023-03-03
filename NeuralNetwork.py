import numpy as np

def sigmoid(x):
    return(np.exp(x)/(1+np.exp(x)))

def layer(x, w, b):
    v=w*x+b
    return(v)

