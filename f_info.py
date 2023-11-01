import numpy as np

def f_gradient(x, Q, rT):
    g_t = 2*Q@x+rT*np.sign(x)
    
    return g_t

def f_value(x, Q, rT):
    f_t = x.T@Q@x+rT*np.sum(np.abs(x))

    return f_t