import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

"""

Función para segmentación por el método de watershed

La segmentación por watershed es un algoritmo basado en la topografía de la imagen,
donde las regiones de interés se consideran como "cuencas" que se llenan de agua desde los "mínimos" (regiones oscuras) hasta que se encuentran con otras cuencas,
formando así los bordes de las regiones segmentadas.

La función `segmentacion_watershed` toma una imagen en escala de grises y un número de iteraciones para las operaciones morfológicas.

La funcion se utiliza con el proposito de obtener o resaltar los bordes de la estrutura que se encuentra en el primer plano de la imagen.

Args:
    imagen (numpy.ndarray): Imagen en escala de grises a segmentar.
    iteraciones (int): Número de iteraciones para las operaciones morfológicas.

"""

def segmentacion_watershed(imagen, iteraciones):
    plt.figure()
    plt.subplot(3, 7, 8)
    plt.title('Imagen Original')
    plt.imshow(imagen, cmap='gray')
    
    # Primero, necesitamos encontrar los marcadores para las regiones
    _, thresh = cv.threshold(imagen, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    plt.subplot(3, 7, 1)
    plt.title('Imagen Umbralizada')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    # Definir kernel para operaciones morfológicas
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=iteraciones)
    plt.subplot(3, 7, 2)
    plt.title('Apertura Morfológica')
    plt.imshow(opening, cmap='gray')
    plt.axis('off')
   
    sure_bg = cv.erode(opening, kernel, iterations=iteraciones)
    plt.subplot(3, 7, 3)
    plt.title('Fondo Seguro')
    plt.imshow(sure_bg, cmap='gray')
    plt.axis('off')

    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    plt.subplot(3, 7, 4)
    plt.title('Transformada de Distancia')
    plt.imshow(dist_transform, cmap='gray')
    plt.axis('off')

    _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    plt.subplot(3, 7, 5)
    plt.title('Primer Plano')
    plt.imshow(sure_fg, cmap='gray')
    plt.axis('off')

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    #Marcadores
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    plt.subplot(3, 7, 6)
    plt.title('Marcadores de esctructuras conectadas')
    plt.imshow(markers, cmap='gray')
    plt.axis('off')

    # Aplicar el algoritmo de watershed
    markers = cv.watershed(cv.cvtColor(sure_bg, cv.COLOR_GRAY2BGR), markers)
    plt.subplot(3, 7, 7)
    plt.title('Marcadores después de watershed')
    plt.imshow(markers, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    imagen[markers == -1] = 255  # Marcar los bordes con blanco

    return imagen

"""

Función para segmentación por el método de Semilla

La segmentación por semilla es un algoritmo que comienza con uno o más píxeles "semilla" y expande la región alrededor de estas semillas
basándose en un criterio de similitud, como la intensidad de píxeles.

La función `segmentacion_semilla` toma una lista de semillas, un umbral de similitud, una imagen en escala de grises y la conectividad (4 o 8).

La funcion apunto a resaltar la estructura que se encuentra en el primer plano de la imagen.

Args:
    seeds (Lista de tuplas): Lista de coordenadas (x, y) de las semillas.
    threshold (int): Umbral de similitud para incluir píxeles en la región segmentada.
    imagen (numpy.ndarray): Imagen en escala de grises a segmentar.
    conectividad (int): Tipo de conectividad (4 o 8) para definir vecinos.

"""

def segmentacion_semilla(seeds, threshold, imagen, conectividad):
    segmented = np.zeros_like(imagen, dtype=np.uint8) #arreglo de 0's para la segmentacion (negro)
    visited = np.zeros_like(imagen, dtype=bool) #arreglo de booleanos para marcar los pixeles visitados
    height, width = imagen.shape # dimensiones de la imagen

    plt.figure()
    
    # Vecinos según conectividad
    if conectividad == 4:
        vecinos = [(-1,0), (1,0), (0,-1), (0,1)]
    else:
        vecinos = [(dx,dy) for dx in [-1,0,1] for dy in [-1,0,1] if not (dx==0 and dy==0)]

    for seed in seeds:
        x, y = seed

        #Se itera por cada una de las semillas definidas
        if not (0 <= x < height and 0 <= y < width):
            print(f"Semilla {seed} fuera de límites (height={height}, width={width})") #se valida que la semilla este dentro de la imagen
            continue

        if visited[x, y]:   #se valida si la semilla x, y ya fue procesada, si ya fue procesada, se mueve a la siguiente 
            continue

        queue = deque()
        queue.append((x, y))
        visited[x, y] = True    # marcar pixel como visitado
        segmented[x, y] = 255   # marcar pixel en la segmentacion blanco
        region_sum = int(imagen[x, y])
        region_count = 1
        while queue:
            px, py = queue.popleft()
            region_mean = region_sum // region_count
            for dx, dy in vecinos:
                nx, ny = px + dx, py + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    if abs(int(imagen[nx, ny]) - region_mean) < threshold:
                        segmented[nx, ny] = 255
                        visited[nx, ny] = True
                        queue.append((nx, ny))
                        region_sum += int(imagen[nx, ny])
                        region_count += 1
        
        plt.subplot(3, 4, seeds.index(seed) + 1)
        plt.title(f'Segmentación para semilla {seed}')
        plt.imshow(segmented, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return segmented

"imagen de 194 x 259 pixeles)"
def segmentar_imagen_monterrey():
    # Cargar la imagen
    raiz = os.getcwd()
    imgPath = os.path.join(raiz, 'imagenes/monterrey.jpg')
    img = cv.imread(imgPath)
    if img is None:
        print('Error al cargar la imagen')
        sys.exit(1)
    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    seeds = [(25, 25), (100, 100), (125, 125), (150, 200)]  # ajustar semilla(s) y umbral según convenga
    threshold = 45
    conectividad = 4
    iteraciones = 3
    
    res_watershed = segmentacion_watershed(gris, iteraciones)

    res_seeded = segmentacion_semilla(seeds, threshold, gris, conectividad)

    desplegar_imagenes(res_watershed, res_seeded)

"imagen de 183 x 275 pixeles)"
def segmentar_imagen_sujeto():
    # Cargar la imagen
    raiz = os.getcwd()
    imgPath = os.path.join(raiz, 'imagenes/sujeto.jpg')
    img = cv.imread(imgPath)
    if img is None:
        print('Error al cargar la imagen')
        sys.exit(1)
    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    seeds = [(25, 50), (50, 100)] # ajustar semilla(s) y umbral según convenga
    threshold = 35
    iteraciones = 5
    conectividad = 4
    res_watershed= segmentacion_watershed(gris, iteraciones)

    res_seeded = segmentacion_semilla(seeds, threshold, gris, conectividad)

    desplegar_imagenes(res_watershed, res_seeded)

"imagen de 148 x 163 pixeles"
def segmentar_imagen_lena():
    # Cargar la imagen
    raiz = os.getcwd()
    imgPath = os.path.join(raiz, 'imagenes/lena.jpg')
    img = cv.imread(imgPath)
    if img is None:
        print('Error al cargar la imagen')
        sys.exit(1)
    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    seeds = [(25, 25), (125, 25), (80, 160)]  # ajustar semilla(s) y umbral según convenga
    threshold = 45
    iteraciones = 4
    conectividad = 8
    res_watershed = segmentacion_watershed(gris, iteraciones)

    res_seeded = segmentacion_semilla(seeds, threshold, gris, conectividad)

    desplegar_imagenes(res_watershed, res_seeded)

def desplegar_imagenes(watershed_img, seeded_img):
    # Mostrar resultados de segmentación
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Segmentacion por Watershed')
    plt.imshow(watershed_img, cmap='gray')
    plt.show()

    plt.subplot(1, 2, 2)
    plt.title('Segmentacion por Semilla')
    plt.imshow(seeded_img, cmap='gray')
    plt.show()

def mostrar_imagen_segmentada(*imagenes):
    for imagen in imagenes:
        if imagen == 'monterrey':
            segmentar_imagen_monterrey()
        elif imagen == 'sujeto':
            segmentar_imagen_sujeto()
        elif imagen == 'lena':
            segmentar_imagen_lena()

"""
Ejercicios de segmentacion usando diferentes metodos: 
    watershed
    Seeds

-Se trata de segmentar tres imagenes diferentes: monterrey.jpg, sujeto.jpg y lena.jpg esto con la finalidad de resaltar las estructuras en primer plano de un fondo complejo.
-Se pueden cambiar las semillas y los umbrales para observar diferentes resultados.
-Las imagenes deben estar en el mismo directorio que este script.
-Cada metodo de segmentacion se visualiza en una ventana diferente.
-Cada vez que se llama a la funcion mostrar_imagen_segmentada con el nombre de la imagen, se ejecuta el proceso de segmentacion para esa imagen.
    el metodo se puede llamar haciendo referencia a las 3 imagenes, lo que mostrara las segmentaciones de las 3 imagenes en ventanas separadas.

Conclusiones:
-Tras analizar los resultados obtenidos se determina que los segmentos mas definidos se obtienen con el metodo de Semillas, seguido del metodo de watershed

Imagenes utilizadas:
monterrey.jpg (https://www.facebook.com/MonterreyEnImagenes/)
sujeto.jpg (https://alvinmint.com/generador-de-historias-aleatorias/)
lena.jpg (https://verne.elpais.com/verne/2019/10/21/articulo/1571661515_258318.html)

"""
if __name__ == "__main__":
    mostrar_imagen_segmentada('monterrey', 'sujeto', 'lena')
