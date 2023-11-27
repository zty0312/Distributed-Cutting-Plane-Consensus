import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

Na = 5

# generate double stochastic matrix
num_degree = np.random.randint(2, Na) 
x = np.random.random((num_degree,num_degree))
rsum = None
csum = None

while (np.any(rsum != 1)) | (np.any(csum != 1)):
    x /= x.sum(0)
    x = x / x.sum(1)[:, np.newaxis]
    rsum = x.sum(1)
    csum = x.sum(0)
print('weight matrix', x)

# generate a directed random graph
weighted_adjacency_matrix = x
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
