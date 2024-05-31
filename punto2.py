# import os
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity
# import seaborn as sns

# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
#             img_path = os.path.join(folder, filename)
#             img = Image.open(img_path)
#             img = img.convert('L')
#             img = img.resize((28, 28))
#             img_array = np.array(img).flatten()
#             images.append(img_array)
#     return images

# def pca_svd(X, n_components):
#     X_mean = np.mean(X, axis=0)
#     X_centered = X - X_mean

#     U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

#     principal_components = Vt[:n_components, :]

#     X_reduced = np.dot(X_centered, principal_components.T)

#     X_reconstructed = np.dot(X_reduced, principal_components) + X_mean

#     return X_reduced, X_reconstructed, Vt

# def show_images_multiple_reconstructions(original_images, reconstructed_images_list, image_shape, components_list):
#     num_images = len(original_images)
#     num_reconstructions = len(reconstructed_images_list)
    
#     plt.figure(figsize=(15, 2 * (num_reconstructions + 1)))
    
#     for i in range(num_images):
#         # Imagen original
#         plt.subplot(num_reconstructions + 1, num_images, i + 1)
#         plt.imshow(original_images[i].reshape(image_shape), cmap='gray')
#         if i == 0: plt.title('Original')
#         plt.axis('off')
        
#         for j in range(num_reconstructions):
#             plt.subplot(num_reconstructions + 1, num_images, (j + 1) * num_images + i + 1)
#             plt.imshow(reconstructed_images_list[j][i].reshape(image_shape), cmap='gray')
#             if i == 0: plt.title(f'd={components_list[j]}')
#             plt.axis('off')
    
#     plt.show()

# def show_eigenvectors(Vt, image_shape):
#     num_components = Vt.shape[0]
#     plt.figure(figsize=(15, 5))
#     for i in range(num_components):
#         plt.subplot(1, num_components, i + 1)
#         plt.imshow(Vt[i].reshape(image_shape), cmap='gray')
#         plt.title(f'Vector {i + 1}')
#         plt.axis('off')
#     plt.show()

# def calculate_mse(original_images, reconstructed_images):
#     mse = np.mean((original_images - reconstructed_images) ** 2)
#     return mse

# def plot_similarity_matrices(original_images, reconstructed_images_list, components_list):
#     num_reconstructions = len(reconstructed_images_list)
    
#     plt.figure(figsize=(15, 5 * num_reconstructions))
    
#     for i, reconstructed_images in enumerate(reconstructed_images_list):
#         similarity_matrix = cosine_similarity(reconstructed_images)
#         plt.subplot(1, num_reconstructions, i + 1)
#         sns.heatmap(similarity_matrix, cmap='viridis')
#         plt.title(f'Similarity Matrix d={components_list[i]}')
#         plt.axis('off')
    
#     plt.show()

# images = load_images_from_folder('datasets_imgs')
# images_matrix = np.array(images)

# components_list = [19, 15, 10, 6, 2]
# reconstructed_images_list = []
# mse_list = []

# for n_components in components_list:
#     _, images_reconstructed, Vt = pca_svd(images_matrix, n_components)
#     reconstructed_images_list.append(images_reconstructed)
#     mse = calculate_mse(images_matrix, images_reconstructed)
#     mse_list.append(mse)

# image_shape = (28, 28)

# show_images_multiple_reconstructions(images_matrix, reconstructed_images_list, image_shape, components_list)

# # Graficar el MSE en función del número de componentes principales
# plt.figure(figsize=(10, 6))
# plt.plot(components_list, mse_list, marker='o')
# plt.xlabel('Número de Componentes Principales (d)')
# plt.ylabel('Error Cuadrático Medio (MSE)')
# plt.title('MSE vs. Número de Componentes Principales')
# plt.grid(True)
# plt.show()

# # Graficar las matrices de similitud
# plot_similarity_matrices(images_matrix, reconstructed_images_list, components_list)

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

    return X_reduced, X_reconstructed, principal_components

def show_images_multiple_reconstructions(original_images, reconstructed_images_list, image_shape, components_list):
    num_images = len(original_images)
    num_reconstructions = len(reconstructed_images_list)
    
    plt.figure(figsize=(15, 2 * (num_reconstructions + 1)))
    
    for i in range(num_images):
        # Imagen original
        plt.subplot(num_reconstructions + 1, num_images, i + 1)
        plt.imshow(original_images[i].reshape(image_shape), cmap='gray')
        if i == 0: plt.title('Original')
        plt.axis('off')
        
        for j in range(num_reconstructions):
            plt.subplot(num_reconstructions + 1, num_images, (j + 1) * num_images + i + 1)
            plt.imshow(reconstructed_images_list[j][i].reshape(image_shape), cmap='gray')
            if i == 0: plt.title(f'd={components_list[j]}')
            plt.axis('off')
    
    plt.show()

def show_eigenvectors(Vt, image_shape):
    num_components = Vt.shape[0]
    plt.figure(figsize=(15, 5))
    for i in range(num_components):
        plt.subplot(1, num_components, i + 1)
        plt.imshow(Vt[i].reshape(image_shape), cmap='gray')
        plt.title(f'Vector {i + 1}')
        plt.axis('off')
    plt.show()

def calculate_mse(original_images, reconstructed_images):
    mse = np.mean((original_images - reconstructed_images) ** 2)
    return mse

def plot_similarity_matrices_separately(original_images, reconstructed_images_list, components_list):
    for i, reconstructed_images in enumerate(reconstructed_images_list):
        similarity_matrix = cosine_similarity(reconstructed_images)
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, cmap='viridis')
        plt.title(f'Similarity Matrix d={components_list[i]}')
        plt.axis('off')
        plt.show()

images = load_images_from_folder('datasets_imgs')
images_matrix = np.array(images)

components_list = [19, 15, 10, 6, 2]
reconstructed_images_list = []
mse_list = []

for n_components in components_list:
    _, images_reconstructed, _ = pca_svd(images_matrix, n_components)
    reconstructed_images_list.append(images_reconstructed)
    mse = calculate_mse(images_matrix, images_reconstructed)
    mse_list.append(mse)

image_shape = (28, 28)

show_images_multiple_reconstructions(images_matrix, reconstructed_images_list, image_shape, components_list)

# Graficar el MSE en función del número de componentes principales
plt.figure(figsize=(10, 6))
plt.plot(components_list, mse_list, marker='o')
plt.xlabel('Número de Componentes Principales (d)')
plt.ylabel('Error Cuadrático Medio (MSE)')
plt.title('MSE vs. Número de Componentes Principales')
plt.grid(True)
plt.show()

# Graficar las matrices de similitud por separado
plot_similarity_matrices_separately(images_matrix, reconstructed_images_list, components_list)
