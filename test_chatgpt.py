import numpy as np

# Assuming M2 is a NumPy array
Na = 5  # Replace with your actual dimension
M2 = np.zeros((Na,Na))  # Example: random matrix for demonstration
print(M2)
# Create a binary matrix where 0 corresponds to M2 != 0 and 1 corresponds to M2 == 0
binary_matrix = (M2 == 0).astype(int)

# Print the binary matrix
print("Binary Matrix:")
print(binary_matrix)