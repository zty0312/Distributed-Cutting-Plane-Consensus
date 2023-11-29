import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

Na = 5

# generate double stochastic matrix
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

def Random_Weighted_Matrix(n, positions): # positions: the position where the weight cannot be zero, data structure: list of tuples
    c = np.random.rand(1, n)
    c = c / np.sum(c)
    weighted_adjacency_matrix = np.zeros((n,n))
    weighted_adjacency_matrix[range(n), range(n)] = c[0, 0] # assign diagnal
    count = 1
    if bool(positions): # The list is not empty
        if len(positions) >= n:
            pos_to_assign = positions[:n] # Choose the first n tuples 
            p_r, p_c = zip(*pos_to_assign) # Separate x and y coordinates into two arrays
            weighted_adjacency_matrix[p_r, p_c] += c[0, count]
            count += 1
            positions = positions[n:] # delete the first n tuples in 'positions' list
        else:
            pos_to_assign = positions[:len(positions)] # Choose all tuples
            p_r, p_c = zip(*pos_to_assign) # Separate x and y coordinates into two arrays
            missing_elements_in_row = [element for element in range(n) if element not in p_r] # Find missing elements
            p_r.extend(missing_elements_in_row) # Append missing elements to the existing list
            missing_elements_in_col = [element for element in range(n) if element not in p_c] # Find missing elements
            p_c.extend(missing_elements_in_col) # Append missing elements to the existing list
            weighted_adjacency_matrix[p_r, p_c] += c[0, count]
            count += 1
            positions = []
    else: # positions is empty
        
    



    for i in range(1, n):
        p_r = GRPdur(n) # row index
        p_c = p_r[::-1] # colum index
        weighted_adjacency_matrix[p_r, p_c] += c[0, i]
    
    return weighted_adjacency_matrix


# generate a directed random graph
weighted_adjacency_matrix = Random_Weighted_Matrix(Na)
print('weighted matrix:\n', weighted_adjacency_matrix)
G = nx.from_numpy_matrix(weighted_adjacency_matrix)
pos = nx.spring_layout(G)  # Layout algorithm (you can choose other layouts)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# assign weights
nodelist = list(range(1, num_nodes + 1))
edgelist = []
for i in nodelist:
    out_neighbour = list(G.successors(i))
    print('current agents out-neighbour', out_neighbour)
    for j in out_neighbour:
        if j == []:
            break
        if i == j:
            edgelist.append((i, j, 0))
        else:
            rand = random.randint(5, 25)
            edgelist.append((i, j, rand))
            edgelist.append((j, i, rand))
print(edgelist)

# visualisation
