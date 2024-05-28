import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo CSV en un DataFrame
df = pd.read_csv('dataset.csv', header=None)

# Eliminar la primera fila y la primera columna
df = df.iloc[1:, 1:]

# Convertir el DataFrame a una matriz de NumPy
matriz = df.values

# print(matriz)

# Plotear la matriz
plt.imshow(matriz, cmap='viridis', aspect='auto')
plt.colorbar()  # Añadir una barra de color para mostrar el gradiente
plt.title('Matriz representada como una grilla')
plt.xlabel('Columnas')
plt.ylabel('Filas')
plt.show()

u, s, v_T = np.linalg.svd(matriz)

# ploteo las matrices en un mismo subplot

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(u, cmap='viridis', aspect='auto')
axs[0].set_title('Matriz U')
axs[0].set_xlabel('Columnas')
axs[0].set_ylabel('Filas')

axs[1].imshow(np.diag(s), cmap='viridis', aspect='auto')
axs[1].set_title('Matriz Sigma')
axs[1].set_xlabel('Columnas')
axs[1].set_ylabel('Filas')

axs[2].imshow(v_T, cmap='viridis', aspect='auto')
axs[2].set_title('Matriz V_T')
axs[2].set_xlabel('Columnas')
axs[2].set_ylabel('Filas')

plt.show()


# Gráfico de barras de los autovalores

plt.bar(range(len(s)), s)
plt.title('Autovalores')
plt.xlabel('Número de autovalor')
plt.ylabel('Valor')
plt.show()


# reduccion de la dimensionalidad de la matriz

k = 2
v_k_T = v_T[:k, :]
s_k = np.diag(s)[:k, :k]
u_k = u[:, :k]

matriz_reducida_2 = u_k @ s_k @ v_k_T

k = 6
v_k_T = v_T[:k, :]
s_k = np.diag(s)[:k, :k]
u_k = u[:, :k]

matriz_reducida_6 = u_k @ s_k @ v_k_T

k = 10 
v_k_T = v_T[:k, :]
s_k = np.diag(s)[:k, :k]
u_k = u[:, :k]

matriz_reducida_10 = u_k @ s_k @ v_k_T

# ploteo las matrices en un mismo subplot

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(matriz_reducida_2, cmap='viridis', aspect='auto')
axs[0].set_title('Matriz reducida a 2 dimensiones')
axs[0].set_xlabel('Columnas')
axs[0].set_ylabel('Filas')

axs[1].imshow(matriz_reducida_6, cmap='viridis', aspect='auto')
axs[1].set_title('Matriz reducida a 6 dimensiones')
axs[1].set_xlabel('Columnas')
axs[1].set_ylabel('Filas')

axs[2].imshow(matriz_reducida_10, cmap='viridis', aspect='auto')
axs[2].set_title('Matriz reducida a 10 dimensiones')
axs[2].set_xlabel('Columnas')
axs[2].set_ylabel('Filas')

plt.show()



# Ploteo de las matrices de similaridad para distintos valores de k y sigma

def euclinean_distance(xi, xj, sigma):
    return np.exp(-np.linalg.norm(xi - xj) / (2 * sigma ** 2))


def similarity_matrix(matriz, sigma):
    n = matriz.shape[0]
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = euclinean_distance(matriz[i, :], matriz[j, :], sigma)
    return s


# Ploteo de las matrices de similaridad para distintos valores de k y sigma

# # uso matrices reducidas a 2 dimensiones para sigma igual a 0.01, 0.1, 1, 10 en un mismo subplot
# sigma_values = [0.01, 0.1, 1, 10]
# k = 2
# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# for i, sigma in enumerate(sigma_values):
#     s = similarity_matrix(matriz_reducida_2, sigma)
#     axs[i].imshow(s, cmap='viridis', aspect='auto')
#     axs[i].set_title(f'Matriz de similaridad con sigma = {sigma}')
#     axs[i].set_xlabel('Columnas')
#     axs[i].set_ylabel('Filas')
# plt.show()

# # uso matrices reducidas a 6 dimensiones para sigma igual a 0.01, 0.1, 1, 10 en un mismo subplot


# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# for i, sigma in enumerate(sigma_values):
#     s = similarity_matrix(matriz_reducida_6, sigma)
#     axs[i].imshow(s, cmap='viridis', aspect='auto')
#     axs[i].set_title(f'Matriz de similaridad con sigma = {sigma}')
#     axs[i].set_xlabel('Columnas')
#     axs[i].set_ylabel('Filas')
# plt.show()

# # uso matrices reducidas a 10 dimensiones para sigma igual a 0.01, 0.1, 1, 10 en un mismo subplot

# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# for i, sigma in enumerate(sigma_values):
#     s = similarity_matrix(matriz_reducida_10, sigma)
#     axs[i].imshow(s, cmap='viridis', aspect='auto')
#     axs[i].set_title(f'Matriz de similaridad con sigma = {sigma}')
#     axs[i].set_xlabel('Columnas')
#     axs[i].set_ylabel('Filas')
# plt.show()


# matriz centrada: 

def center_matrix(matriz):
    media_columnas = np.mean(matriz, axis=0, keepdims=True)
    matriz_cent = matriz - media_columnas
    return matriz_cent


# ploteo de la matriz centrada

matriz_cent = center_matrix(matriz)
plt.imshow(matriz_cent, cmap='viridis', aspect='auto')
plt.colorbar()  # Añadir una barra de color para mostrar el gradiente
plt.title('Matriz centrada')
plt.xlabel('Columnas')
plt.ylabel('Filas')
plt.show()

s_cent, u_cent, v_T_cent = np.linalg.svd(matriz_cent)

# ploteo las matrices en un mismo subplot

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(u_cent, cmap='viridis', aspect='auto')
axs[0].set_title('Matriz U centrada')
axs[0].set_xlabel('Columnas')
axs[0].set_ylabel('Filas')

axs[1].imshow(np.diag(s_cent), cmap='viridis', aspect='auto')
axs[1].set_title('Matriz Sigma centrada')
axs[1].set_xlabel('Columnas')
axs[1].set_ylabel('Filas')

axs[2].imshow(v_T_cent, cmap='viridis', aspect='auto')
axs[2].set_title('Matriz V_T centrada')
axs[2].set_xlabel('Columnas')
axs[2].set_ylabel('Filas')

plt.show()



