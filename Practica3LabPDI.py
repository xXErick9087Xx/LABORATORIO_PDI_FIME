# =========================
# EJERCICIO 1
# =========================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# CAMBIAR NOMBRE DE IMAGEN
# =========================
imagen1 = "Imagen1.jpg"

# Cargar imagen
img = cv2.imread(imagen1)

# Convertir BGR a RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mostrar imagen
plt.imshow(img_rgb)
plt.title("Imagen original")
plt.axis("off")
plt.show()

# Convertir imagen a lista de pixeles
pixeles = img_rgb.reshape((-1,3))

R = pixeles[:,0]
G = pixeles[:,1]
B = pixeles[:,2]

# Grafica 3D del espacio RGB
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(R, G, B, c=pixeles/255.0, marker='.')

ax.set_xlabel("Rojo")
ax.set_ylabel("Verde")
ax.set_zlabel("Azul")

plt.title("Espacio de color RGB")
plt.show()

# =========================
# EJERCICIO 2
# =========================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CAMBIAR NOMBRE DE IMAGEN
# =========================
imagen2 = "Imagen2.jpg"

# Cargar imagen
img = cv2.imread(imagen2)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convertir a matriz de pixeles
pixeles = img_rgb.reshape((-1,3))
pixeles = np.float32(pixeles)

def aplicar_kmeans(k):

    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.2)

    ret, labels, centros = cv2.kmeans(
        pixeles,
        k,
        None,
        criterio,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centros = np.uint8(centros)

    resultado = centros[labels.flatten()]
    imagen_segmentada = resultado.reshape(img_rgb.shape)

    return imagen_segmentada

# Aplicar clustering
k2 = aplicar_kmeans(2)
k4 = aplicar_kmeans(4)
k6 = aplicar_kmeans(6)

# Mostrar resultados
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(k2)
plt.title("K = 2")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(k4)
plt.title("K = 4")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(k6)
plt.title("K = 6")
plt.axis("off")

plt.show()

# =========================
# EJERCICIO 3
# =========================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CAMBIAR NOMBRE DE IMAGEN
# =========================
imagen3 = "Imagen5.jpg"

# Cargar imagen
img = cv2.imread(imagen3)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preparar pixeles
pixeles = img_rgb.reshape((-1,3))
pixeles = np.float32(pixeles)

# Número de clusters
k = 3

criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.2)

ret, labels, centros = cv2.kmeans(
    pixeles,
    k,
    None,
    criterio,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centros = np.uint8(centros)

resultado = centros[labels.flatten()]
segmentada = resultado.reshape(img_rgb.shape)

# Mostrar resultados
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmentada)
plt.title("Segmentación con K-means (k=3)")
plt.axis("off")

plt.show()