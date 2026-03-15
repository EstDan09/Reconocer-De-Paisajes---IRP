# ============================================================
# test.py  –  Clasificación de imágenes usando el modelo entrenado
# ============================================================
# Este script carga el modelo generado por train.py y clasifica
# todas las imágenes de la carpeta "test_images" como PAISAJE
# o NO PAISAJE.
#
# Convención de nombres para medir exactitud automáticamente:
#   paisaje_foto.jpg   → se espera que sea PAISAJE
#   nopaisaje_foto.jpg → se espera que sea NO PAISAJE
#   cualquier_otro.jpg → se clasifica igual pero no cuenta para el porcentaje
#
# Los resultados también se guardan en "Resultados_prueba.txt"
# ============================================================

import cv2
import numpy as np
import os

def extract_features(image_path):
    """
    Extrae las mismas 3 características que usa train.py.
    Es exactamente la misma función para garantizar consistencia
    entre el entrenamiento y la clasificación.
    """
    # Leer la imagen desde disco
    img = cv2.imread(image_path)

    # Si no se pudo leer (formato no soportado, ruta inválida, etc.), salir
    if img is None:
        return None

    # Reducir la imagen a 320x240 para uniformizar el tamaño
    img = cv2.resize(img, (320, 240))

    # Convertir a espacio de color HSV para facilitar la detección de colores
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]

    # Dividir la imagen en 3 franjas horizontales
    top    = hsv[0:h//3, :]        # tercio superior  → cielo
    middle = hsv[h//3:2*h//3, :]   # tercio del medio → zona de transición
    bottom = hsv[2*h//3:h, :]      # tercio inferior  → suelo, vegetación, edificios

    # Rango de colores que corresponden al cielo azul despejado
    lower_sky = np.array([90, 30, 50])
    upper_sky = np.array([130, 255, 255])

    # Rango de colores que corresponden a vegetación verde
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Feature 1: fracción del tercio superior que es cielo azul (0 = nada, 1 = todo)
    sky_mask_top = cv2.inRange(top, lower_sky, upper_sky)
    sky_ratio_top = np.count_nonzero(sky_mask_top) / sky_mask_top.size

    # Feature 2: fracción del tercio medio que NO es cielo (alta = zona no-cielo, típico de paisajes)
    sky_mask_middle = cv2.inRange(middle, lower_sky, upper_sky)
    non_sky_ratio_middle = 1 - (np.count_nonzero(sky_mask_middle) / sky_mask_middle.size)

    # Feature 3: fracción del tercio inferior que es vegetación o estructuras grises
    green_mask_bottom = cv2.inRange(bottom, lower_green, upper_green)

    s = bottom[:, :, 1]  # canal Saturation
    v = bottom[:, :, 2]  # canal Value (brillo)
    # Píxeles grises: poca saturación (no coloridos) y alto brillo (no oscuros)
    gray_building_mask = ((s < 35) & (v > 80)).astype(np.uint8) * 255

    # Combinar vegetación + estructuras grises en una sola máscara de "suelo"
    ground_mask = cv2.bitwise_or(green_mask_bottom, gray_building_mask.astype(np.uint8))
    ground_ratio_bottom = np.count_nonzero(ground_mask) / ground_mask.size

    return np.array([sky_ratio_top, non_sky_ratio_middle, ground_ratio_bottom])


def load_model(model_path="Modelo reconocedor de paisajes.txt"):
    """
    Lee el archivo generado por train.py y devuelve el vector de
    promedios y el vector de desviaciones estándar.
    """
    mean = None
    std = None

    with open(model_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Promedio:"):
                mean = np.array(eval(line.split(":", 1)[1].strip()))
            elif line.startswith("Desviacion estandar:"):
                std = np.array(eval(line.split(":", 1)[1].strip()))

    return mean, std


def classify(features, mean, std, k=1.0):
    """
    Clasifica una imagen como PAISAJE (True) o NO PAISAJE (False).

    Estrategia de puntuación:
      - Cada feature dentro del rango típico del modelo suma 1 punto.
      - Tener algo de cielo arriba suma 1 punto extra.
      - Tener algo de suelo abajo suma 1 punto extra.
      - Si acumula 3 o más puntos de 5 posibles → PAISAJE.

    El rango típico se define como: promedio ± k * desviación_estándar
    """
    sky_ratio_top, non_sky_ratio_middle, ground_ratio_bottom = features

    # Calcular límites inferior y superior para cada feature
    lower = mean - k * std
    upper = mean + k * std

    score = 0

    # +1 si la proporción de cielo está dentro del rango aprendido durante el entrenamiento
    if lower[0] <= sky_ratio_top <= upper[0]:
        score += 1

    # +1 si la zona media está dentro del rango aprendido
    if lower[1] <= non_sky_ratio_middle <= upper[1]:
        score += 1

    # +1 si la proporción de suelo está dentro del rango aprendido
    if lower[2] <= ground_ratio_bottom <= upper[2]:
        score += 1

    # +1 si hay al menos un 10% de cielo arriba (regla mínima de paisaje)
    if sky_ratio_top >= 0.10:
        score += 1

    # +1 si hay al menos un 10% de suelo/vegetación abajo (regla mínima de paisaje)
    if ground_ratio_bottom >= 0.10:
        score += 1

    # Se considera PAISAJE si cumple al menos 3 de los 5 criterios
    return score >= 3


def infer_expected_label(filename):
    """
    Deduce la etiqueta esperada a partir del nombre del archivo:
      - Empieza con "paisaje_"   → True  (es un paisaje)
      - Empieza con "nopaisaje_" → False (no es un paisaje)
      - Cualquier otro nombre    → None  (sin etiqueta, no cuenta para el porcentaje)
    """
    name = filename.lower()

    if name.startswith("paisaje_"):
        return True
    elif name.startswith("nopaisaje_"):
        return False
    else:
        return None


if __name__ == "__main__":
    model_path   = "Modelo reconocedor de paisajes.txt"
    test_folder  = "test_images"
    results_file = "Resultados_prueba.txt"

    # Verificar que el modelo entrenado existe
    if not os.path.exists(model_path):
        print(f"No existe el archivo '{model_path}'. Ejecutá train.py primero.")
        exit()

    # Verificar que la carpeta de prueba existe
    if not os.path.exists(test_folder):
        print(f"No existe la carpeta '{test_folder}'")
        exit()

    # Cargar el modelo (promedio y desviación estándar aprendidos en entrenamiento)
    mean, std = load_model(model_path)

    if mean is None or std is None:
        print("No se pudo cargar el modelo.")
        exit()

    total_with_label = 0  # imágenes con etiqueta conocida (para calcular exactitud)
    correct = 0           # predicciones correctas

    # Abrir archivo de resultados para guardar todo lo que se imprime
    with open(results_file, "w", encoding="utf-8") as out:
        out.write("Resultados de clasificación\n")
        out.write("=" * 50 + "\n\n")

        for filename in os.listdir(test_folder):
            path = os.path.join(test_folder, filename)

            # Ignorar subdirectorios
            if not os.path.isfile(path):
                continue

            # Extraer las 3 características de la imagen
            features = extract_features(path)

            if features is None:
                print(f"No se pudo leer: {filename}")
                out.write(f"{filename} -> No se pudo leer\n")
                continue

            # Clasificar usando el modelo cargado
            prediction = classify(features, mean, std, k=1.0)
            predicted_label = "PAISAJE" if prediction else "NO PAISAJE"

            # Intentar inferir la etiqueta correcta desde el nombre del archivo
            expected = infer_expected_label(filename)

            line = f"{filename} -> {predicted_label} -> {features.tolist()}"
            print(line)
            out.write(line + "\n")

            # Si el archivo tiene etiqueta, evaluar si la predicción fue correcta
            if expected is not None:
                expected_label = "PAISAJE" if expected else "NO PAISAJE"
                is_correct = (prediction == expected)

                total_with_label += 1
                if is_correct:
                    correct += 1

                out.write(f"  Esperado: {expected_label}\n")
                out.write(f"  Resultado: {'CORRECTO' if is_correct else 'INCORRECTO'}\n")
            else:
                out.write("  Esperado: No indicado por nombre de archivo\n")

            out.write("\n")

        # Mostrar el porcentaje de aciertos final (solo sobre imágenes con etiqueta)
        if total_with_label > 0:
            accuracy = (correct / total_with_label) * 100
            summary = f"Porcentaje de aciertos: {accuracy:.2f}% ({correct}/{total_with_label})"
            print("\n" + summary)
            out.write(summary + "\n")
        else:
            msg = "No hubo imágenes con etiqueta inferible. Renombrá las imágenes con 'paisaje_' o 'nopaisaje_' para medir exactitud."
            print("\n" + msg)
            out.write(msg + "\n")