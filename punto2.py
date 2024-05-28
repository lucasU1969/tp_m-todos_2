import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = img.convert('L')  # Convertir a escala de grises
            img_array = np.array(img).flatten()  # Convertir a vector
            images.append(img_array)
    return images

def pca(X, n_components):
    # Centrar los datos
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Calcular la matriz de covarianza
    covariance_matrix = np.cov(X_centered, rowvar=False)
    
    # Calcular los eigenvalores y eigenvectores
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Ordenar los eigenvectores por eigenvalores decrecientes
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Seleccionar los primeros n_components eigenvectores
    principal_components = eigenvectors[:, :n_components]
    
    # Proyectar los datos a los componentes principales
    X_reduced = np.dot(X_centered, principal_components)
    
    # Reconstrucción aproximada
    X_reconstructed = np.dot(X_reduced, principal_components.T) + X_mean
    
    return X_reduced, X_reconstructed, principal_components

# Especifica la ruta de la carpeta
folder_path = 'datasets_imgs'

# Carga todas las imágenes de la carpeta
images = load_images_from_folder(folder_path)

# Convertir la lista de imágenes a una matriz numpy
images_matrix = np.array(images)

# Aplicar PCA
n_components = 10  # Cambia este valor según lo necesites
images_reduced, images_reconstructed, _ = pca(images_matrix, n_components)

# Mostrar las imágenes originales y las reconstruidas
def show_images(original_images, reconstructed_images, image_shape):
    num_images = len(original_images)
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        # Imagen original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(image_shape), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Imagen reconstruida
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructed_images[i].reshape(image_shape), cmap='gray')
        plt.title('Reconstruida')
        plt.axis('off')
    
    plt.show()

# Tamaño original de las imágenes (debe coincidir con target_size)
image_shape = (28, 28)

# Mostrar imágenes originales y reconstruidas
show_images(images_matrix, images_reconstructed, image_shape)
