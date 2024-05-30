import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = img.convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img).flatten()
            images.append(img_array)
    return images

def pca_svd(X, n_components):
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    principal_components = Vt[:n_components, :]

    X_reduced = np.dot(X_centered, principal_components.T)

    X_reconstructed = np.dot(X_reduced, principal_components) + X_mean

    return X_reduced, X_reconstructed, Vt

def show_images(original_images, reconstructed_images, image_shape):
    num_images = len(original_images)
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        # Imagen original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(image_shape), cmap='gray')
        # plt.title('Original')
        plt.axis('off')

        # Imagen reconstruida
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructed_images[i].reshape(image_shape), cmap='gray')
        # plt.title('Reconstruida')
        plt.axis('off')

    plt.show()

# Mostrar los eigenvectores
def show_eigenvectors(Vt, image_shape):
    num_components = Vt.shape[0]
    plt.figure(figsize=(15, 5))
    for i in range(num_components):
        plt.subplot(1, num_components, i + 1)
        plt.imshow(Vt[i].reshape(image_shape), cmap='gray')
        plt.title(f'Vector {i + 1}')
        plt.axis('off')
    plt.show()

images = load_images_from_folder('datasets_imgs')
images_matrix = np.array(images)

n_components = 19
images_reduced, images_reconstructed, Vt = pca_svd(images_matrix, n_components)

image_shape = (28, 28)

# show_images(images_matrix, images_reconstructed, image_shape)
# show_eigenvectors(Vt[:n_components], image_shape)




def plot_similarity_matrix(similarity_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title(title)
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.show()

# Valores de d a utilizar
d_values = [2, 6, 10, 50, 100]

for d in d_values:
    print(f"\nResultados para d = {d}:")

    # Aplicar PCA utilizando SVD
    images_reduced, images_reconstructed, Vt = pca_svd(images_matrix, d)

    # Calcular la matriz de similaridad utilizando la similitud del coseno
    similarity_matrix = cosine_similarity(images_reduced)

    # Mostrar la matriz de similaridad
    plot_similarity_matrix(similarity_matrix, f"Similarity Matrix (d={d})")

    # Opcional: Imprimir la matriz de similaridad
    print(f"Matriz de similaridad (d={d}):")
    print(similarity_matrix)
