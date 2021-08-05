import numpy as np

def alphabetize(x,y):
    if x.get_name() > y.get_name():
        return 1
    return -1

def abs_mean(values):
    """Compute the mean of the absolute values of a set of numbers.
    For computing the stopping condition for training neural nets"""
    return np.mean(np.abs(values))

def sigmoid(value):
    return 1/(np.exp(-value) + 1)

def sigmoid_derivative(value):
    return np.exp(-value)/np.power(np.exp(-value) + 1, 2)
