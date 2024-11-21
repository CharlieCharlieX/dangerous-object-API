from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import os
from azure.storage.blob import BlobServiceClient
import uuid

app = FastAPI()

# Ruta al repositorio y al modelo
repo_path = 'yolov5'
model_path = os.path.join(repo_path, 'best.pt')

# Cargar el modelo entrenadp
model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')

# Extensiones de archivo que serán eliminadas cada vez que se realice un request al API
VIDEO_EXTENSIONS = ['.avi', '.mp4', '.mkv', '.mov', '.flv', '.wmv']

# Configuración de Azure Blob Storage
AZURE_CONNECTION_STRING = "CONNECTION_STRING_HERE"  
BLOB_CONTAINER_NAME = "object-videos-analized"

# Crear cliente de servicio de Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

#Función para eliminar todos los Blobs del contenedor de Microsoft Azure
def clean_blob_container(container_name: str):
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs()
    for blob in blobs:
        container_client.delete_blob(blob.name)
        print(f"Blob eliminado: {blob.name}")

#Función para subir un archivo al contenedor de Microsoft Azure
def upload_blob(file_path: str, blob_name: str, container_name: str):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Archivo subido: {blob_name}")

#Función para eliminar todos los archivos con extensión de video que existen en el directorio
def remove_video_files():
    """Eliminar archivos de video en el directorio actual"""
    current_directory = os.getcwd() 
    for filename in os.listdir(current_directory):
        if any(filename.endswith(ext) for ext in VIDEO_EXTENSIONS):
            file_path = os.path.join(current_directory, filename)
            try:
                os.remove(file_path)
                print(f"Archivo eliminado: {file_path}")
            except Exception as e:
                print(f"Error al eliminar el archivo {file_path}: {e}")

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    file_path = file.filename

    unique_id = f"{str(uuid.uuid4())}.avi"

    #Eliminar cualquier video en el directorio
    remove_video_files()

    # Limpiar el contenedor de blobs
    clean_blob_container(BLOB_CONTAINER_NAME)

    # Guardar el archivo en el directorio
    with open(file_path, "wb") as f:
        # Escribir el contenido del archivo recibido
        content = await file.read()
        f.write(content)

    # Procesar el video
    output_path = "output.avi"
        
    cap = cv2.VideoCapture(file_path)
        
    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        return {"error": "No se pudo abrir el video."}
        
    # Obtener las dimensiones del video de entrada
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    # Crear un VideoWriter para guardar el resultado con las dimensiones del video de entrada
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    # Procesar el video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realizar la predicción (detección de objetos)
        results = model(frame_rgb)

        # Renderizar las predicciones en la imagen
        img = results.render()[0]  # Obtener la imagen con predicciones

        # Convertir la imagen renderizada de RGB a BGR para OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Guardar el frame procesado en el video de salida
        out.write(img_bgr)

    cap.release()
    out.release()

    upload_blob(output_path, unique_id, BLOB_CONTAINER_NAME)

    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{unique_id}"
    return {"video_url": blob_url}






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
