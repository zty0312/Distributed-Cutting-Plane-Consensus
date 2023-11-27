import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
import matplotlib.pyplot as plt

Na = 5

# generate double stochastic matrix
def generate_doubly_stochastic_matrix_with_zeros(n, num_zeros_per_row, num_zeros_per_column):
    # Generate a random doubly stochastic matrix
    doubly_stochastic_matrix = np.random.rand(n, n)
    doubly_stochastic_matrix /= doubly_stochastic_matrix.sum(axis=1)[:, np.newaxis]
    doubly_stochastic_matrix /= doubly_stochastic_matrix.sum(axis=0)

    # Introduce additional zeros
    for _ in range(num_zeros_per_row):
        # Randomly select entries excluding diagonal
        row_indices = np.random.choice(np.delete(np.arange(n), np.arange(0, n, n+1)), size=num_zeros_per_column, replace=False)
        doubly_stochastic_matrix[:, row_indices] = 0

    return doubly_stochastic_matrix


def generate_random_doubly_stochastic_matrix(n):
    # Generate random non-negative matrix
    random_matrix = np.random.rand(n, n)

    # Normalize rows to ensure the sum of each row is 1
    random_matrix /= random_matrix.sum(axis=1)[:, np.newaxis]

    # Normalize columns to ensure the sum of each column is 1
    random_matrix /= random_matrix.sum(axis=0)

    return random_matrix

def insert_row_and_column(matrix):
    idx_zero = np.random.randint(1,num_degree)
    # Generate a new row
    new_row = np.zeros((1, matrix.shape[1]))
    # Insert the new row
    matrix = np.insert(matrix, idx_zero-1, new_row, axis=0)
    # Generate a new column
    new_column = np.zeros((1, matrix.shape[0]))
    # Insert the new row and new column
    matrix = np.insert(matrix, idx_zero-1, new_column, axis=1)
    print('enlarged matrix\n', matrix)
    matrix[idx_zero-1, idx_zero-1] = 1

    return matrix

# Specify the size of the doubly stochastic matrix
matrix_size = Na

# Specify the number of zeros per row and column
num_zeros_per_row = 1
num_zeros_per_column = 2

# Generate a doubly stochastic matrix with more than one zero entry per row and column
result_matrix = generate_doubly_stochastic_matrix_with_zeros(matrix_size, num_zeros_per_row, num_zeros_per_column)
print('new random matrix\n', result_matrix)
num_degree = np.random.randint(2, Na) 
weighted_adjacency_matrix = generate_random_doubly_stochastic_matrix(num_degree)

for i in range (Na-num_degree):
    weighted_adjacency_matrix = insert_row_and_column(weighted_adjacency_matrix)

print('weighted adjacency matrix is\n', weighted_adjacency_matrix)

# generate a directed random graph
G = nx.from_numpy_array(weighted_adjacency_matrix)

# visualisation
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, with_labels=True, node_size=700, node_color='skyblue', font_size=8)
plt.show()
