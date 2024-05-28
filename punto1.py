# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# # Cargar el dataset excluyendo la primera fila y la primera columna
# dataset = pd.read_csv('dataset.csv', index_col=0).iloc[1:, 1:]
# X = dataset.values

# # Imprimir la matriz X leída del archivo CSV
# print("Matriz X leída del archivo CSV:")

# # Centrar la matriz X
# scaler = StandardScaler(with_std=False)
# X_centered = scaler.fit_transform(X)

# # Realizar SVD
# U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

# def compute_similarity_matrix(X, sigma):
#     n = X.shape[0]
#     similarity_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             distance = np.linalg.norm(X[i] - X[j])
#             similarity_matrix[i, j] = np.exp(-(distance**2) / (2 * sigma**2))
#     return similarity_matrix

# # Definir los valores de d
# dimensions = [2, 6, 10, X.shape[1]]
# sigma = 1.0  # Se puede ajustar según el caso

# for d in dimensions:
#     Z = U[:, :d] @ np.diag(S[:d])
#     similarity_matrix = compute_similarity_matrix(Z, sigma)
#     plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
#     plt.title(f'Similarity Matrix for d = {d}')
#     plt.colorbar()
#     plt.show()

# # Varianza explicada
# explained_variance = np.cumsum(S**2) / np.sum(S**2)

# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Explained Variance by Number of Components')
# plt.grid(True)
# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Cargar el dataset excluyendo la primera fila y la primera columna
dataset = pd.read_csv('dataset.csv', index_col=0).iloc[1:, 1:]
X = dataset.values

# Centrar la matriz X
scaler = StandardScaler(with_std=False)
X_centered = scaler.fit_transform(X)

# Realizar SVD
U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

def compute_similarity_matrix(Z, sigma):
    n = Z.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance = np.linalg.norm(Z[i] - Z[j])
            similarity_matrix[i, j] = np.exp(-(distance**2) / (2 * sigma**2))
    return similarity_matrix

# Definir los valores de d
dimensions = [2, 6, 10, X.shape[1]]
sigma = 1.0  # Se puede ajustar según el caso

for d in dimensions:
    Z = U[:, :d] @ np.diag(S[:d])
    similarity_matrix = compute_similarity_matrix(Z, sigma)
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.title(f'Similarity Matrix for d = {d}')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.colorbar()
    plt.show()

# Varianza explicada
explained_variance = np.cumsum(S**2) / np.sum(S**2)

plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.grid(True)
plt.show()
