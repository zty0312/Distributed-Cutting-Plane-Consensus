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
               pos_to_assign = positions[:len(acceptable_postions)] # Choose all tuples
               p_r, p_c = zip(*pos_to_assign) # Separate x and y coordinates into two arrays
               p_r = np.array(p_r)
               p_c = np.array(p_c)               
               missing_elements_in_row = [element for element in range(n) if element not in p_r] # Find missing elements
               p_r = np.append(p_r, missing_elements_in_row, axis=0) # Append missing elements to the existing list
               missing_elements_in_col = [element for element in range(n) if element not in p_c] # Find missing elements
               p_c = np.append(p_c, missing_elements_in_col, axis=0) # Append missing elements to the existing list
               weighted_adjacency_matrix[p_r, p_c] += c[0, count]
               count += 1
               positions = []
        else:
            weighted_adjacency_matrix = Set_fully_connected(n)
            print("fully connected graph")
            return weighted_adjacency_matrix
        
    for i in range(Na-count+1, Na):
        p_r = GRPdur(n) # row index
        p_c = p_r[::-1] # colum index
        weighted_adjacency_matrix[p_r, p_c] += c[0, i]
    
    return weighted_adjacency_matrix


# generate a directed random graph
non_zero_positions = [(0,3), (0,4), (1,2), (2,1),(3,0),(3,2), (4,0), (4,3)]
weighted_adjacency_matrix = Random_Weighted_Matrix(Na, non_zero_positions)
print('weighted matrix:\n', weighted_adjacency_matrix)
print('col sum:', np.sum(weighted_adjacency_matrix, axis=1))
print('row sum:', np.sum(weighted_adjacency_matrix, axis=0))
G = nx.from_numpy_array(weighted_adjacency_matrix)


# visualisation
pos = nx.spring_layout(G)  # Layout algorithm (you can choose other layouts)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
