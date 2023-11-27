import numpy as np
import networkx as nx
import f_info
from f_info import*
from DCP_C import idx_zero
def creat_params(Na, dim_a, max_iter,upb,lwb):
    # creat matrix Q and r
    A = np.zeros((dim_a,Na))
    for i in range(Na):
        random_colum = np.random.randint(1,5,dim_a)
        A[:,i] = random_colum
    
    random_matrix = np.random.randint(1,6,size=(dim_a,dim_a))
    Q = np.zeros((dim_a,Na*dim_a))
    S,_ = np.linalg.qr(random_matrix)
    for i in range(Na):
        Q_i = np.linalg.inv(S)@np.diag(A[:,i])@S
        Q[:,i*dim_a:(i+1)*dim_a] = Q_i

    r = np.random.normal(0, 1, size=(1,Na))

    # set initial value
    initial_tem = np.zeros((dim_a,Na))
    for i in range(Na):
        current_upb = upb[:,i]
        current_lwb = lwb[:,i]
        for j in range(dim_a):
            initial_tem[j,i] = np.random.randint(current_lwb,current_upb,1)
    initial_sum = np.sum(initial_tem,1)
    agent_initial = 1/Na*initial_sum
    y_initial = agent_initial

    # calculate initial g and f
    x = np.zeros((Na*dim_a,Na*max_iter))
    g = np.zeros((Na*dim_a,Na*max_iter))
    f = np.zeros((Na,Na*max_iter))
    Q_sum = np.zeros((dim_a,dim_a))
    for i in range(Na):
        x[i*dim_a:(i+1)*dim_a,i*max_iter] = agent_initial
        Q_current = Q[:,i*dim_a:(i+1)*dim_a]
        Q_sum = Q_sum+Q_current
        r_t_current = r[:,i]
        g_current = f_info.f_gradient(agent_initial,Q_current,r_t_current)
        g[i*dim_a:(i+1)*dim_a,i*max_iter] = g_current
        f_current = f_info.f_value(agent_initial,Q_current,r_t_current)
        f[i:(i+1)*dim_a,i*max_iter] = f_current

    x_star = np.zeros((dim_a,1))
    f_val = x_star.T @ Q_sum @ x_star + np.sum(r) * np.linalg.norm(x_star)

    agent = [0]*11
    agent = {'x':x, 
             'y_initial':y_initial, 
             'g':g,   
             'f':f,
             'Q':Q, 
             'rT':r,
             'x_star':x_star,
             'f_star':f_val,
             'lower_bound': lwb,
             'upper_bound': upb,
             'dimension':dim_a,
             'num_agents':Na}
    
    return agent

def agent_slice(agent, agent_idx, current_step, max_iter, Ga):
    x = agent.x
    g = agent.g
    f = agent.f
    Q = agent.Q
    r = agent.rT
    dim = agent.dimension
    topo = Ga[:,agent_idx]
    N_Ga = np.size(Ga, 0)
    Na = agent.num_agents

    current_x = x[:,agent_idx*max_iter:(agent_idx+1)*max_iter]
    current_g = g[:,agent_idx*max_iter:(agent_idx+1)*max_iter]
    current_f = f[:,agent_idx*max_iter:(agent_idx+1)*max_iter]
    for i in range(N_Ga):
        if topo(i) != 0:
            current_x[i*dim:(i+1)*dim,current_step] = x[i*dim:(i+1)*dim,i*max_iter+current_step+1]
            current_g[i*dim:(i+1)*dim,current_step] = g[i*dim:(i+1)*dim,i*max_iter+current_step+1] 
            current_f[i,current_step] = f[i,i*max_iter+current_step] 
        else:
            current_x[i*dim:(i+1)*dim,current_step] = current_x[i*dim:(i+1)*dim,current_step]
            current_g[i*dim:(i+1)*dim,current_step] = current_g[i*dim:(i+1)*dim,current_step]
            current_f[i,current_step] = current_f[i,current_step]        

    
    agent = [0]*6
    current_agent = {'x_memory':current_x, 
                    'g_memory':current_g,   
                    'f_memory':current_f,
                    'Q':Q[:,agent_idx*dim:(agent_idx+1)*dim], 
                    'rT':r[:,agent_idx],
                    'topo':topo,
                    }

    return current_agent


def Creat_Graph(Na,idx_pre):
    num_nodes = Na  # Adjust as needed
    # Generate a random doubly stochastic adjacency matrix
    adjacency_matrix = np.random.rand(num_nodes, num_nodes)
    row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
    col_sums = adjacency_matrix.sum(axis=0, keepdims=True)
    doubly_stochastic_matrix = adjacency_matrix / row_sums / col_sums

    # Create a graph from the adjacency matrix
    doubly_stochastic_graph = nx.Graph(doubly_stochastic_matrix)
    binary_matrix = (doubly_stochastic_graph == 0).astype(int)
    idx_zero = idx_zero+binary_matrix
    idx_reset = (idx_zero == idx_pre).astype(int)
    idx_zero[idx_reset == 1] = 0


    return doubly_stochastic_graph


def Graph_update(Na, T):
    zero_again = 1

    while zero_again != 0:
        idx_pre = idx_zero
        Ga_0 = Creat_Graph(Na, idx_pre)
        row, col = np.where(idx_zero == T)
        pos = list(zip(row, col))
        if len(row) > 0:
            # Find row and column indices where Ga_0 is equal to 0
            row_new, col_new = np.where(Ga_0 == 0)
            pos_new = list(zip(row_new, col_new))
            
            # Check for common positions between pos_new and pos
            common_positions = set(pos_new).intersection(pos)
            
            if common_positions:
                zero_again = len(common_positions)
                idx_zero = idx_pre

        elif np.any(np.diag(Ga_0) == 0):
            idx_zero = idx_pre
        else:
            zero_again = 0
    
    Ga = Ga_0
    return Ga

     





