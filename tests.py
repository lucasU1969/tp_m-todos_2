import numpy as np

def restar_promedio_columnas(matriz):
    matriz = np.array(matriz)
    promedios = np.mean(matriz, axis=0)
    matriz_centrada = matriz - promedios
    matriz_centrada = matriz_centrada.tolist()
    return matriz_centrada

# Ejemplo de uso
matriz = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print("Matriz original:")
for fila in matriz:
    print(fila)

matriz_centrada = restar_promedio_columnas(matriz)

print("\nMatriz centrada (promedio de cada columna restado):")
for fila in matriz_centrada:
    print(fila)
