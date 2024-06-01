import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo CSV en un DataFrame
df = pd.read_csv('dataset.csv', header=None)

# Eliminar la primera fila y la primera columna
df = df.iloc[1:, 1:]

# Convertir el DataFrame a una matriz de NumPy
matriz = df.values

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




# Ploteo de las matrices de similaridad para distintos valores de k y sigma

def euclinean_distance(xi, xj, sigma):
    return np.exp(-np.linalg.norm(xi - xj)**2 / (2 * sigma ** 2))


def similarity_matrix(matriz, sigma):
    n = matriz.shape[0]
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = euclinean_distance(matriz[i, :], matriz[j, :], sigma)
    return s


# matriz centrada: 
def center_matrix(matriz):
    # """
    # Centra la matriz restando la media de cada fila a los elementos que se encuentran en la fila
    # :param matriz: matriz a centrar
    # :return: matriz centrada
    # """
    # return matriz - np.mean(matriz, axis=1).reshape(-1, 1)
    # restar a cada posición el promedio de la columna en la que se encuentra
    return matriz - np.mean(matriz, axis=0)



# Centrando la matriz
matriz_cent = center_matrix(matriz)

# Ploteo de la matriz centrada
plt.imshow(matriz_cent, cmap='viridis', aspect='auto')
plt.colorbar()  # Añadir una barra de color para mostrar el gradiente
plt.title('Matriz centrada')
plt.xlabel('Columnas')
plt.ylabel('Filas')
plt.show()

# Realizar SVD
u_cent, s_cent, v_T_cent = np.linalg.svd(matriz_cent)

# Asegurar que s_cent sea bidimensional para la visualización
sigma_cent = np.zeros((u_cent.shape[1], v_T_cent.shape[0]))
np.fill_diagonal(sigma_cent, s_cent)

# Ploteo de las matrices en un mismo subplot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(u_cent, cmap='viridis', aspect='auto')
axs[0].set_title('Matriz U centrada')
axs[0].set_xlabel('Columnas')
axs[0].set_ylabel('Filas')

axs[1].imshow(sigma_cent, cmap='viridis', aspect='auto')
axs[1].set_title('Matriz Sigma centrada')
axs[1].set_xlabel('Columnas')
axs[1].set_ylabel('Filas')

axs[2].imshow(v_T_cent, cmap='viridis', aspect='auto')
axs[2].set_title('Matriz V_T centrada')
axs[2].set_xlabel('Columnas')
axs[2].set_ylabel('Filas')

plt.show()

# ahora hay que recortar cada matriz para quedarnos con las primeras 2 columnas, luego las primeras 6 y por último las primeras 10 y plotearlas junto a la matriz original

# Recortar las matrices
u_cent_2 = u_cent[:, :2]
sigma_cent_2 = sigma_cent[:2, :2]
v_T_cent_2 = v_T_cent[:2, :]
matriz_reducida_cent_2 = u_cent_2 @ sigma_cent_2 @ v_T_cent_2

u_cent_6 = u_cent[:, :6]
sigma_cent_6 = sigma_cent[:6, :6]
v_T_cent_6 = v_T_cent[:6, :]
matriz_reducida_cent_6 = u_cent_6 @ sigma_cent_6 @ v_T_cent_6

u_cent_10 = u_cent[:, :10]
sigma_cent_10 = sigma_cent[:10, :10]
v_T_cent_10 = v_T_cent[:10, :]
matriz_reducida_cent_10 = u_cent_10 @ sigma_cent_10 @ v_T_cent_10

# Ploteo de las matrices en un mismo subplot
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(matriz, cmap='viridis', aspect='auto')
axs[0].set_title('Matriz original')
axs[0].set_xlabel('Columnas')
axs[0].set_ylabel('Filas')

axs[1].imshow(matriz_reducida_cent_2, cmap='viridis', aspect='auto')
axs[1].set_title('Matriz reducida a 2 dimensiones')
axs[1].set_xlabel('Columnas')
axs[1].set_ylabel('Filas')

axs[2].imshow(matriz_reducida_cent_6, cmap='viridis', aspect='auto')
axs[2].set_title('Matriz reducida a 6 dimensiones')
axs[2].set_xlabel('Columnas')
axs[2].set_ylabel('Filas')

axs[3].imshow(matriz_reducida_cent_10, cmap='viridis', aspect='auto')
axs[3].set_title('Matriz reducida a 10 dimensiones')
axs[3].set_xlabel('Columnas')
axs[3].set_ylabel('Filas')

plt.show()

# # ahora voy a plotear los resultados de hacer la matriz centrada por v_T_cent_k para k = 2, 6 y 10

# # Ploteo de las matrices en un mismo subplot
# fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# hay que trasponer la matriz v_T_cent_k para cada k
v_cent_2 = v_T_cent[:2, :].T
v_cent_6 = v_T_cent[:6, :].T
v_cent_10 = v_T_cent[:10, :].T


Z_cent_2 = matriz_reducida_cent_2 @ v_cent_2
Z_cent_6 = matriz_reducida_cent_6 @ v_cent_6
Z_cent_10 = matriz_reducida_cent_10 @ v_cent_10

Z_cent_2  = u_cent_2 @ sigma_cent_2

# axs[0].imshow(matriz, cmap='viridis', aspect='auto')
# axs[0].set_title('Matriz original')
# axs[0].set_xlabel('Columnas')
# axs[0].set_ylabel('Filas')

# axs[1].imshow(Z_cent_2, cmap='viridis', aspect='auto')
# axs[1].set_title('Matriz reducida a 2 dimensiones')
# axs[1].set_xlabel('Columnas')
# axs[1].set_ylabel('Filas')

# axs[2].imshow(Z_cent_6, cmap='viridis', aspect='auto')
# axs[2].set_title('Matriz reducida a 6 dimensiones')
# axs[2].set_xlabel('Columnas')
# axs[2].set_ylabel('Filas')

# axs[3].imshow(Z_cent_10, cmap='viridis', aspect='auto')
# axs[3].set_title('Matriz reducida a 10 dimensiones')
# axs[3].set_xlabel('Columnas')
# axs[3].set_ylabel('Filas')

# plt.show()


# # # ahora hay que plotear las matrices de similaridad para distintos valores de k y sigma

# uso matrices reducidas a 2 dimensiones para sigma igual a 0.01, 0.1, 1, 10 en un mismo subplot
sigma_values = [0.01, 0.1, 1, 10]
k = 2
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for i, sigma in enumerate(sigma_values):
    s = similarity_matrix(Z_cent_2, sigma)
    axs[i].imshow(s, cmap='viridis', aspect='auto')
    axs[i].set_title(f'Matriz de similaridad con sigma = {sigma}')
    axs[i].set_xlabel('Columnas')
    axs[i].set_ylabel('Filas')
plt.title('Matriz de similaridad con sigma = 0.01, 0.1, 1, 10 y d = 2')
plt.show()

# uso matrices reducidas a 6 dimensiones para sigma igual a 0.01, 0.1, 1, 10 en un mismo subplot
k = 6
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for i, sigma in enumerate(sigma_values):
    s = similarity_matrix(Z_cent_6, sigma)
    axs[i].imshow(s, cmap='viridis', aspect='auto')
    axs[i].set_title(f'Matriz de similaridad con sigma = {sigma}')
    axs[i].set_xlabel('Columnas')
    axs[i].set_ylabel('Filas')
plt.title('Matriz de similaridad con sigma = 0.01, 0.1, 1, 10 y d = 6')
plt.show()

# uso matrices reducidas a 10 dimensiones para sigma igual a 0.01, 0.1, 1, 10 en un mismo subplot
k = 10
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for i, sigma in enumerate(sigma_values):
    s = similarity_matrix(Z_cent_10, sigma)
    axs[i].imshow(s, cmap='viridis', aspect='auto')
    axs[i].set_title(f'Matriz de similaridad con sigma = {sigma}')
    axs[i].set_xlabel('Columnas')
    axs[i].set_ylabel('Filas')
plt.title('Matriz de similaridad con sigma = 0.01, 0.1, 1, 10 y d = 10')
plt.show()



# def PCA(matrix, d): 
#     # Centrando la matriz
#     matrix_cent = center_matrix(matrix)

#     # Realizar SVD
#     u_cent, s_cent, v_T_cent = np.linalg.svd(matrix_cent)

#     # Asegurar que s_cent sea bidimensional para la visualización
#     sigma_cent = np.zeros((u_cent.shape[1], v_T_cent.shape[0]))
#     np.fill_diagonal(sigma_cent, s_cent)

#     # Recortar las matrices
#     u_cent_d = u_cent[:, :d]
#     sigma_cent_d = sigma_cent[:d, :d]
#     v_T_cent_d = v_T_cent[:d, :]
#     matrix_reducida_cent_d = u_cent_d @ sigma_cent_d @ v_T_cent_d

#     # Ploteo de las matrices en un mismo subplot
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))

#     axs[0].imshow(matrix, cmap='viridis', aspect='auto')
#     axs[0].set_title('Matriz original')
#     axs[0].set_xlabel('Columnas')
#     axs[0].set_ylabel('Filas')

#     axs[1].imshow(matrix_reducida_cent_d, cmap='viridis', aspect='auto')
#     axs[1].set_title(f'Matriz reducida a {d} dimensiones')
#     axs[1].set_xlabel('Columnas')
#     axs[1].set_ylabel('Filas')

#     plt.show()

#     # ahora hay que plotear las matrices de similaridad para distintos valores de k y sigma

#     # uso matrices reducidas a d dimensiones para sigma igual a 0.01, 0.1, 1, 10 en un mismo subplot
#     sigma_values = [0.01, 0.1, 1, 10]
#     fig, axs = plt.subplots(1, 4, figsize=(20, 5))

#     for i, sigma in enumerate(sigma_values):
#         s = similarity_matrix(matrix_reducida_cent_d, sigma)
#         axs[i].imshow(s, cmap='viridis', aspect='auto')
#         axs[i].set_title(f'Matriz de similaridad con sigma = {sigma}')
#         axs[i].set_xlabel('Columnas')
#         axs[i].set_ylabel('Filas')
#     plt.title(f'Matriz de similaridad con sigma = 0.01, 0.1, 1, 10 y d = {d}')
#     plt.show()


# PCA(matriz, 2)
# PCA(matriz, 6)
# PCA(matriz, 10)


# ahora tengo que plotear los vectores propios asociados a las k primeras componentes principales con cada coordenada como una barra en un gráfico de barras
# saco los vectores propios de la matriz V_T, matriz en la que los vectores están dispuestos como filas
# gráfico de barras de la primera fila de V_T y la segunda fila de V_T
# Ploteo de los vectores propios en subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].bar(range(len(v_T_cent_2[0, :])), v_T_cent_2[0, :], color='green')
axs[0].set_title('Primer vector propio')
axs[0].set_xlabel('Componente')
axs[0].set_ylabel('Valor')
axs[1].bar(range(len(v_T_cent_2[1, :])), v_T_cent_2[1, :], color='green')
axs[1].set_title('Segundo vector propio')
axs[1].set_xlabel('Componente')
axs[1].set_ylabel('Valor')
plt.show()

# ploteo los datos recortados luego de multiplicar por la v recortada con d = 2

# posibes cmap = 'viridis', 'plasma', 'inferno', 'magma', 'cividis'



def plot_2d_colormap(matrix):
    """
    Grafica cada observación en una matriz de tamaño n x 2 en el plano 2D con un colormap.

    Parámetros:
    matrix (np.ndarray): Matriz de tamaño n x 2 donde n son el número de observaciones y 2 son las features.
    """
    # Comprobar si la matriz tiene las dimensiones correctas
    if matrix.shape[1] != 2:
        raise ValueError("La matriz debe tener exactamente 2 columnas (features)")

    # Crear un colormap basado en el número de observaciones
    n_observations = matrix.shape[0]
    colors = np.linspace(0, 1, n_observations)

    # Crear la figura y el eje
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(matrix[:, 0], matrix[:, 1], c=colors, cmap='viridis', s=50, alpha=0.7, edgecolor='k')

    # Añadir la barra de color
    plt.colorbar(scatter, label='Observación index')

    # Etiquetas y título
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Mediciones en R2 con Colormap')
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()


plot_2d_colormap(Z_cent_2)

# plt.scatter(Z_cent_2[:, 0], Z_cent_2[:, 1],color='red' ,cmap='viridis')
# plt.title('Datos recortados con d = 2')
# plt.xlabel('Componente 1')
# plt.ylabel('Componente 2')
# plt.show()


#calcular la norma de Frobenius entre las matrices de similaridad de la matriz original y la matriz reducida a 2 dimensiones

def frobenius_norm(matrix1, matrix2):
    """
    Calcula la norma de Frobenius entre dos matrices.

    Parámetros:
    matrix1 (np.ndarray): Primera matriz.
    matrix2 (np.ndarray): Segunda matriz.

    Retorna:
    float: Norma de Frobenius entre las dos matrices.
    """
    return np.linalg.norm(matrix1 - matrix2, ord='fro')


# # Calcular las matrices de similaridad
# s_original = similarity_matrix(matriz, 1)
# s_reducida = similarity_matrix(Z_cent_2, 1)

# # Calcular la norma de Frobenius
# frobenius = frobenius_norm(s_original, s_reducida)
# print(f'(d = 2) La norma de Frobenius entre las matrices de similaridad es: {frobenius}')

# #calcular la norma de Frobenius entre las matrices de similaridad de la matriz original y la matriz reducida a 6 dimensiones

# # Calcular las matrices de similaridad
# s_original = similarity_matrix(matriz, 1)
# s_reducida = similarity_matrix(Z_cent_6, 1)

# # Calcular la norma de Frobenius
# frobenius = frobenius_norm(s_original, s_reducida)
# print(f'(d = 6) La norma de Frobenius entre las matrices de similaridad es: {frobenius}')

# #calcular la norma de Frobenius entre las matrices de similaridad de la matriz original y la matriz reducida a 10 dimensiones

# # Calcular las matrices de similaridad
# s_original = similarity_matrix(matriz, 1)
# s_reducida = similarity_matrix(Z_cent_10, 1)

# # Calcular la norma de Frobenius
# frobenius = frobenius_norm(s_original, s_reducida)
# print(f'(d = 10) La norma de Frobenius entre las matrices de similaridad es: {frobenius}')



# ahora voy a calcular la norma de frobenius entre la matriz original y las matrices reducidas

# Calcular la norma de Frobenius entre la matriz original y la matriz reducida a 2 dimensiones
frobenius = frobenius_norm(matriz, matriz_reducida_cent_2)
print(f'(d = 2) La norma de Frobenius entre las matrices es: {frobenius}')

# Calcular la norma de Frobenius entre la matriz original y la matriz reducida a 6 dimensiones
frobenius = frobenius_norm(matriz, matriz_reducida_cent_6)
print(f'(d = 6) La norma de Frobenius entre las matrices es: {frobenius}')

# Calcular la norma de Frobenius entre la matriz original y la matriz reducida a 10 dimensiones
frobenius = frobenius_norm(matriz, matriz_reducida_cent_10)
print(f'(d = 10) La norma de Frobenius entre las matrices es: {frobenius}')