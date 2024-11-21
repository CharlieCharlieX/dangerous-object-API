from pathlib import Path
import torch
import cv2
import os


# Ruta al repositorio y al modeloz
repo_path = 'C:/Users/Usuario/OneDrive/Escritorio/Trabajos/Weapon_Object_Detection/yolov5'
model_path = os.path.join(repo_path, 'best.pt')

# Cargar el modelo YOLOv5
model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')

# Leer el video
cap = cv2.VideoCapture('C:/Users/Usuario/OneDrive/Escritorio/Trabajos/Weapon_Object_Detection/yolov5/data/images/Berreta_9mm_Pistol.mp4')

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Obtener las dimensiones del video de entrada
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ruta para guardar el video generado
output_path = 'C:/Users/Usuario/OneDrive/Escritorio/Trabajos/Weapon_Object_Detection/output.avi'

# Crear un VideoWriter para guardar el resultado con las dimensiones del video de entrada
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

# Procesar el video sin mostrar cada frame (para evitar sobrecargar Colab)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen de BGR (OpenCV) a RGB (YOLO)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar la predicción (detección de objetos)
    results = model(frame_rgb)

    # Renderizar las predicciones en la imagen
    img = results.render()[0]  # Obtener la imagen con predicciones

    # Convertir la imagen renderizada de RGB a BGR para OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Guardar el frame procesado en el video de salida
    out.write(img_bgr)
 

# Liberar los recursos
cap.release()
out.release()

