import os
from PIL import Image
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
    return images

def convert_to_matrix(images):
    return np.array(images)

# Especifica la ruta de la carpeta
folder_path = 'datasets_imgs'

# Carga todas las imágenes de la carpeta
images = load_images_from_folder(folder_path)

# Convierte la lista de imágenes a una matriz numpy
images_matrix = convert_to_matrix(images)

print(images_matrix.shape)  # Esto imprimirá las dimensiones de la matriz resultante



#################################################################3


def unzip_and_load_images(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return load_images_from_folder(extract_to)

# Especifica la ruta del archivo zip y la carpeta de extracción
zip_path = 'dataset_imagenes1.zip'
extract_to = 'datasets_imgs'

# Descomprimir y cargar todas las imágenes del archivo zip
images = unzip_and_load_images(zip_path, extract_to)

# Convertir la lista de imágenes a una matriz numpy
images_matrix = np.array(images)

# Aplicar SVD
svd = TruncatedSVD(n_components=50)  # Cambia n_components según lo necesites
svd.fit(images_matrix)
images_svd = svd.transform(images_matrix)

print("Shape of original image matrix:", images_matrix.shape)
print("Shape of reduced image matrix:", images_svd.shape)
