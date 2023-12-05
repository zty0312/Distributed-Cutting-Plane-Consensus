import numpy as np
import networkx as nx
import f_info
from itertools import combinations
from f_info import*

def create_params(Na, dim_a, max_iter, upb, lwb):
    # creat matrix Q and r
    A = np.zeros((dim_a, Na))
    for i in range(Na):
        random_colum = np.random.randint(1, 5, dim_a)
        A[:,i] = random_colum
    
    random_matrix = np.random.randint(1, 6, size=(dim_a,dim_a))
    Q = np.zeros((dim_a, Na*dim_a))
    S,_ = np.linalg.qr(random_matrix)
    for i in range(Na):
        Q_i = S.T@np.diag(A[:,i])@S # Q_i & Q have different address
        Q[:, i*dim_a:(i+1)*dim_a] = Q_i
    

    r = np.random.normal(0, 1, size=(1,Na))

    # set initial value
    initial_tem = np.zeros((dim_a, Na))
    for i in range(Na):
        current_upb = upb[:, i]
        current_lwb = lwb[:, i]
        for j in range(dim_a):
            initial_tem[j, i] = np.random.randint(current_lwb, current_upb, 1)

    initial_sum = np.sum(initial_tem, 1)
    agent_initial = 1/Na*initial_sum
    y_initial = agent_initial # same address

    # calculate initial g and f
    x = np.zeros((Na*dim_a, Na*max_iter))
    g = np.zeros((Na*dim_a, Na*max_iter))
    f = np.zeros((Na, Na*max_iter))
    Q_sum = np.zeros((dim_a, dim_a))
    for i in range(Na):
        x[i*dim_a:(i+1)*dim_a, i*max_iter] = agent_initial
        Q_current = Q[:, i*dim_a:(i+1)*dim_a]
        Q_sum = Q_sum + Q_current
        r_t_current = r[:, i]
        g_current = f_info.f_gradient(agent_initial, Q_current, r_t_current)
        g[i*dim_a:(i+1)*dim_a, i*max_iter] = g_current
        f_current = f_info.f_value(agent_initial, Q_current, r_t_current)
        f[i, i*max_iter] = f_current

    x_star = np.zeros((dim_a,1))
    f_val = x_star.T @ Q_sum @ x_star + np.sum(r) * np.linalg.norm(x_star)

    agent = [0]*11
    agent = {'x': x, 
             'y_initial': y_initial, 
             'g': g,   
             'f': f,
             'Q': Q, 
             'rT': r,
             'x_star': x_star,
             'f_star': f_val,
             'lower_bound': lwb,
             'upper_bound': upb,
             'dimension': dim_a}
    
    return agent

def agent_slice(agent, agent_idx, current_step, max_iter, Ga):
    x = agent['x']
    g = agent['g']
    f = agent['f']
    Q = agent['Q']
    r = agent['rT']
    dim = agent['dimension']
    topo = Ga[:,agent_idx]
    N_Ga = np.size(Ga, 0)

    current_x = x[:, agent_idx*max_iter:(agent_idx+1)*max_iter]
    current_g = g[:, agent_idx*max_iter:(agent_idx+1)*max_iter]
    current_f = f[:, agent_idx*max_iter:(agent_idx+1)*max_iter]
    for i in range(N_Ga):
        if topo[i] != 0:
            current_x[i*dim:(i+1)*dim, current_step-1] = x[i*dim:(i+1)*dim, i*max_iter+current_step-1]
            current_g[i*dim:(i+1)*dim, current_step-1] = g[i*dim:(i+1)*dim, i*max_iter+current_step-1] 
            current_f[i, current_step-1] = f[i, i*max_iter+current_step-1] 
        else:
            current_x[i*dim:(i+1)*dim, current_step-1] = current_x[i*dim:(i+1)*dim, current_step-2]
            current_g[i*dim:(i+1)*dim,current_step-1] = current_g[i*dim:(i+1)*dim, current_step-2]
            current_f[i,current_step-1] = current_f[i,current_step-1]        

    
    agent = [0]*6
    current_agent = {'x_memory': current_x, 
                    'g_memory': current_g,   
                    'f_memory': current_f,
                    'Q':Q[:, agent_idx*dim:(agent_idx+1)*dim], 
                    'rT':r[:, agent_idx],
                    'topo': topo,
                    }

    return current_agent

def Set_fully_connected(n):
    full_matrix = np.random.rand(n, n) # Generate random non-negative matrix
    full_matrix /= full_matrix.sum(axis=1)[:, np.newaxis] # Normalize rows to ensure the sum of each row is 1
    full_matrix /= full_matrix.sum(axis=0) # Normalize columns to ensure the sum of each column is 1
    return full_matrix

def find_longest_combination_in_different_rows_and_columns(positions):
    longest_combination = []

    for r in range(2, len(positions) + 1):  # Start from pairs, up to the total number of positions
        for combination in combinations(positions, r): # Check if each point is in a different row and column   
            if all(combination[i][0] != combination[j][0] and combination[i][1] != combination[j][1]
                   for i in range(r) for j in range(i + 1, r)):
                longest_combination = combination

    return longest_combination

def GRPdur(n): # generate index
    p = np.arange(0, n)  # Start with Identity permutation

    for k in range(n, 0, -1):
        r = np.random.randint(0, k)  # random integer between 1 and k
        t = p[k-1]
        p[k-1] = p[r-1]  # Swap(p(r),p(k)).
        p[r-1] = t

    return p

def create_Graph(n,idx_pre, T):
    positions_row, positions_col = np.where(idx_pre == T) # extract positions where the weights cannot be zero again
    # Generate a random doubly stochastic adjacency matrix
    c = np.random.rand(1, n)
    c = c / np.sum(c)
    weighted_adjacency_matrix = np.zeros((n,n))
    weighted_adjacency_matrix[range(n), range(n)] = c[0, 0] # assign diagnal
    if bool(np.size(positions_row) > 0): # there are positions that cannot be zero
        positions = list(zip(positions_row, positions_col))
        count = 1
        while bool(positions): # The list is not empty
            acceptable_postions = find_longest_combination_in_different_rows_and_columns(positions) # find positions in different row and col
            if count < n:
                if len(acceptable_postions) >= n:
                    pos_to_assign = acceptable_postions[:n] # Choose the first n tuples 
                    p_r, p_c = zip(*pos_to_assign) # Separate x and y coordinates into two arrays
                    p_r = np.array(p_r)
                    p_c = np.array(p_c)
                    weighted_adjacency_matrix[p_r, p_c] += c[0, count]
                    count += 1
                    set_a = set(positions) # delete used tuples in 'positions' list
                    set_b = set(acceptable_postions)
                    result = set_a.symmetric_difference(set_b)
                    positions = list(result)
                else:
                    pos_to_assign = acceptable_postions # Choose all tuples
                    p_r, p_c = zip(*pos_to_assign) # Separate x and y coordinates into two arrays
                    p_r = np.array(p_r)
                    p_c = np.array(p_c)               
                    missing_elements_in_row = [element for element in range(n) if element not in p_r] # Find missing elements
                    missing_elements_in_col = [element for element in range(n) if element not in p_c] # Find missing elements
                    p_r = np.append(p_r, missing_elements_in_row, axis=0) # Append missing elements to the existing list
                    p_c = np.append(p_c, missing_elements_in_col, axis=0) # Append missing elements to the existing list
                    weighted_adjacency_matrix[p_r, p_c] += c[0, count]
                    count += 1
                    positions = []
            else:
                weighted_adjacency_matrix = Set_fully_connected(n)
                print("fully connected graph")
                return weighted_adjacency_matrix
        
        for i in range(n-count-1, n):
            p_r = GRPdur(n) # row index
            p_c = p_r[::-1] # colum index
            weighted_adjacency_matrix[p_r, p_c] += c[0, i]

    else: # randomly selected doubly stochastic matrix without restrictions
        for i in range(1, n):
            p_r = GRPdur(n) # row index
            p_c = p_r[::-1] # colum index
            weighted_adjacency_matrix[p_r, p_c] += c[0, i]

    # update idx_pre
    idx_current = np.copy(idx_pre) # initialise idx_current
    idx_zero_new_row, idx_zero_new_col = np.where(weighted_adjacency_matrix == 0) # find idx of zero entries
    idx_current[idx_zero_new_row, idx_zero_new_col] += 1 # update idx of zero entries
    idx_reset_row, idx_reset_col = np.where(idx_pre == idx_current)
    idx_current[idx_reset_row, idx_reset_col] = 0
    
    return weighted_adjacency_matrix, idx_current

def update_graph(k, Na, T, idx_zero):
    if k == 2:
        Ga = 1/Na*np.ones((Na, Na))
    else:
        Ga, idx_zero = create_Graph(Na, idx_zero, T)
    return Ga, idx_zero

    #print('Graph is:\n', Ga)
    #print('zero position matrix is:\n', idx_zero)

def communication(Na, agent_all, k, max_iter, Ga):
    for i in range(Na):
        update = agent_slice(agent_all, i, k-1, max_iter, Ga)
        update_x = update['x_memory']
        update_g = update['g_memory']
        update_f = update['f_memory']
        x = agent_all['x']
        g = agent_all['g']
        f = agent_all['f']
        x[:, i*max_iter+k-1] = update_x[:, k-1]
        g[:, i*max_iter+k-1] = update_g[:, k-1]
        f[:, i*max_iter+k-1] = update_f[:, k-1]
    return agent_all
    

     





