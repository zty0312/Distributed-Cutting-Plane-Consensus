import numpy as np
import networkx as nx
import cvxpy as cp
import networkx as nx
import copy
import multiprocessing
import matplotlib.pyplot as plt
import create_data as cd
import f_info as f
import prepare_algorithm as prep

# initialisation
Na = 3
dim_a = 2
T = 3
max_iter = 5
agent_upb = 10*np.ones((dim_a, Na))
agent_lwb = -10*np.ones((dim_a, Na))
agent_all = cd.create_params(Na, dim_a, max_iter, agent_upb, agent_lwb)

idx_zero = np.zeros((Na, Na))
x = 0
y_pre = np.kron(agent_all['y_initial'], np.ones((1, Na)))

def DCP_T(i, dim, k):
    # refine cutting plane
    current_agent = cd.agent_slice(agent_all, i, k-1, max_iter, Ga)
    memory_x = current_agent['x_memory']
    topo = current_agent['topo']
    current_upb = agent_upb[:, i]
    current_lwb = agent_lwb[:, i]
    current_Q = current_agent['Q']
    current_rT = current_agent['rT']
    tilde_gjm_i, tilde_fjm_i, xim = prep.refine_cutting_plane(k, current_agent, i, dim, Na)

    # find next query point
    for j in range(Na):
        x +=  topo[j] * memory_x[j*dim:(j+1)*dim, k-2]
    
    print('x:', x)
    return x



for k in range(2, max_iter+1):
    # create graph
    Ga, idx_zero = cd.update_graph(k, Na, T, idx_zero)

    # communication
    agent_all = cd.communication(Na, agent_all, k, max_iter, Ga)
    for i in range(Na):
        DCP_T(i, dim_a, k)

    




