# ============================================================
# train.py  –  Entrenamiento del reconocedor de paisajes
# ============================================================
# Este script lee todas las imágenes de la carpeta "imagenes",
# extrae 3 características de color de cada una y calcula el
# promedio y la desviación estándar de esas características.
# Esos valores se guardan en un archivo .txt que luego usa
# test.py para clasificar imágenes nuevas.
# ============================================================

import cv2
import numpy as np
import os

def extract_features(image_path):
    """
    Recibe la ruta de una imagen y devuelve un vector con 3 valores
    que describen el contenido visual de la imagen en términos de color.
    """
    # Leer la imagen desde disco
    img = cv2.imread(image_path)

    # Si no se pudo leer (formato no soportado, ruta inválida, etc.), salir
    if img is None:
        return None

    # Reducir la imagen a 320x240 para que todas tengan el mismo tamaño
    # y el procesamiento sea más rápido
    img = cv2.resize(img, (320, 240))

    # Convertir de BGR (formato OpenCV) a HSV (Hue-Saturation-Value)
    # HSV facilita detectar colores específicos como el azul del cielo
    # o el verde de la vegetación independientemente de la iluminación
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]

    # Dividir la imagen en 3 franjas horizontales:
    # los paisajes suelen tener cielo arriba y suelo/vegetación abajo
    top    = hsv[0:h//3, :]        # tercio superior  → donde suele estar el cielo
    middle = hsv[h//3:2*h//3, :]   # tercio del medio → zona de transición
    bottom = hsv[2*h//3:h, :]      # tercio inferior  → suelo, vegetación, edificios

    # Rango de tonos azules en HSV que corresponden al cielo despejado
    # H: 90-130 (azul), S: 30-255, V: 50-255
    lower_sky = np.array([90, 30, 50])
    upper_sky = np.array([130, 255, 255])

    # Rango de tonos verdes en HSV que corresponden a vegetación
    # H: 35-85 (amarillo-verde a verde), S y V mínimos para evitar blancos/grises
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # --- Feature 1: proporción de cielo en el tercio superior ---
    # Crea una máscara donde los píxeles azules (cielo) son blancos y el resto negro
    sky_mask_top = cv2.inRange(top, lower_sky, upper_sky)
    # Calcula qué fracción de píxeles del tercio superior es cielo (0.0 a 1.0)
    sky_ratio_top = np.count_nonzero(sky_mask_top) / sky_mask_top.size

    # --- Feature 2: proporción NO cielo en el tercio del medio ---
    # Si hay mucho cielo en el medio, probablemente NO es un paisaje típico
    sky_mask_middle = cv2.inRange(middle, lower_sky, upper_sky)
    non_sky_ratio_middle = 1 - (np.count_nonzero(sky_mask_middle) / sky_mask_middle.size)

    # --- Feature 3: proporción de suelo/vegetación/estructuras en el tercio inferior ---
    # Detectar píxeles verdes (pasto, árboles, arbustos)
    green_mask_bottom = cv2.inRange(bottom, lower_green, upper_green)

    # Detectar píxeles grises (cemento, asfalto, edificios)
    # Criterio: saturación baja (gris, no colorido) y brillo alto (no oscuro)
    s = bottom[:, :, 1]  # canal Saturation
    v = bottom[:, :, 2]  # canal Value (brillo)
    gray_building_mask = ((s < 35) & (v > 80)).astype(np.uint8) * 255

    # Combinar la máscara verde y la máscara gris en una sola máscara de "suelo"
    ground_mask = cv2.bitwise_or(green_mask_bottom, gray_building_mask.astype(np.uint8))
    # Calcular qué fracción del tercio inferior es suelo/vegetación/edificio
    ground_ratio_bottom = np.count_nonzero(ground_mask) / ground_mask.size

    # Devolver el vector de 3 características
    return [sky_ratio_top, non_sky_ratio_middle, ground_ratio_bottom]


def process_folder(folder_path, output_txt="Individuales.txt"):
    """
    Procesa todas las imágenes de una carpeta, extrae sus características
    y las guarda en un archivo .txt. Devuelve una matriz con todos los vectores.
    """
    vectors = []

    with open(output_txt, "w", encoding="utf-8") as f:
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            # Ignorar subdirectorios, solo procesar archivos
            if not os.path.isfile(path):
                continue

            features = extract_features(path)

            if features is not None:
                vectors.append(features)
                # Guardar en el .txt el nombre del archivo y su vector de características
                f.write(f"{filename}: {features}\n")
                print(f"Procesada: {filename} -> {features}")
            else:
                print(f"No se pudo leer: {filename}")

    # Convertir la lista de vectores a una matriz de NumPy (filas = imágenes, columnas = features)
    return np.array(vectors)


if __name__ == "__main__":
    # Carpeta con las imágenes de entrenamiento (todas deben ser paisajes)
    folder = "imagenes"

    if not os.path.exists(folder):
        print(f"No existe la carpeta '{folder}'")
        exit()

    # Extraer características de todas las imágenes
    vectors = process_folder(folder)

    if len(vectors) > 0:
        # Calcular el promedio y la desviación estándar de cada feature
        # Esto define el "rango típico" de un paisaje para cada característica
        mean_vector = np.mean(vectors, axis=0)
        std_vector  = np.std(vectors, axis=0)

        # Guardar el modelo en un archivo de texto
        # Este archivo es el que usa test.py para clasificar imágenes nuevas
        with open("Modelo reconocedor de paisajes.txt", "w", encoding="utf-8") as f:
            f.write(f"Promedio: {mean_vector.tolist()}\n")
            f.write(f"Desviacion estandar: {std_vector.tolist()}\n")

        print("\nModelo generado correctamente.")
        print("Promedio:", mean_vector)
        print("Desviación estándar:", std_vector)
    else:
        print("No se procesaron imágenes válidas.")