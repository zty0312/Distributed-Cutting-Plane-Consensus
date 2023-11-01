import numpy as np
import networkx as nx
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import creat_data as cd
from creat_data import *

import f_info as f
from f_info import *

# initialisation
Na = 3
dim_a = 3
T = 3
max_iter = 10
agent_upb = 10*np.ones((dim_a, Na))
agent_lwb = -10*np.ones((dim_a, Na))
agent_all = cd.creat_params(Na, dim_a, max_iter, agent_upb, agent_lwb)

idx_zero = np.zeros((Na, Na))
x = 0
y_pre = np.kron(agent_all['y_initial'], np.ones((1, Na)))

for i in range(2, max_iter+1):
    if i == 2:
        Ga = 1/Na*np.ones((Na, Na))
    else:
        Ga = cd.Graph_updat(Na, T)

    print(Ga)

