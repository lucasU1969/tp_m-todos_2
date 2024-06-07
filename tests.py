import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dataset.csv', header=None)

df = df.iloc[1:, 1:]

matriz = df.values

matriz_cent = matriz - np.mean(matriz, axis=0)

u, s, v_T = np.linalg.svd(matriz_cent, full_matrices=False)

def truncate_svd(u, s, v_T, d):
    return u[:, :d], s[:d], v_T[:d, :]
d = 2
u_2, s_2, v_T_2 = truncate_svd(u, s, v_T, d)
v_2 = v_T_2.T
Z_cent_2 = matriz @ v_2

with open('y.txt', 'r') as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]

y = np.array(lines, dtype=float)
y_cent = y - np.mean(y)

def pseudo_inverse(matriz):
    u, s, v_T = np.linalg.svd(matriz, full_matrices=False)
    s_inv = np.diag(1/s)
    return v_T.T @ s_inv @ u.T

u, s, v_T = np.linalg.svd(matriz_cent, full_matrices=False)
d = 2
u_2, s_2, v_T_2 = truncate_svd(u, s, v_T, d)
v_2 = v_T_2.T
Z_cent_2 = matriz @ v_2
pseudo_inv = pseudo_inverse(Z_cent_2)
beta = pseudo_inv @ y_cent

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = beta[0] * X + beta[1] * Y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.scatter(Z_cent_2[:, 0], Z_cent_2[:, 1], y_cent, c=y_cent, cmap='coolwarm')
ax.set_title('Plano de regresión')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('y')
plt.show()