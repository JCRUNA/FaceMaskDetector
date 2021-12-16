# importar paquetes necesarios

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

#en este script importamos los paquetes anteriores para cargar nuestro modelo de deteccion de barbijos
# y preprocesar la imagen
#opencv lo usamos para preprocesar la imagen

#creamos la linea de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="ruta a la imagen de entrada")
ap.add_argument("-f","--face",type=str,default="face-mask-detector", help="ruta al directorio del face_Detector")
ap.add_argument("-m","--model",type=str,default="mask_detector.model", help="ruta al modelo")
ap.add_argument("-c","--confidence",type=float,default=0.5, help="probabilidad minima para filtrar detecciones debiles")
args = vars(ap.parse_args())

# Ahora cargaremos nuestro detector de rostros del disco
print("Cargando modelo de deteccion de rostros del disco...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"]) #ruta de acceso al model de deteccion de rostros
weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"]) #ruta de acceso a los pesos del detector de rostros
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("cargando el modelo de deteccion de rostros...")
model = load_model(args["model"])

#con nuestro modelo de deeplearning en memoria, el siguiente paso es 
#Cargar y preprocesar la imagen

image = cv2.imread(args["image"])
orig = image.copy() #creo una copia de la imagen
(h, w) = image.shape[:2]  #obtengo el ancho y alto de la imagen

#construyo blob de imagen. Redimensiono la imagen a 300 x 300 y hago la resta de la media para cada canal RGB
blob = cv2.dnn.blobFromImage(image,1.0,(300,300), (104.0, 177.0, 123.0))

#paso el blob por la red  y obtengo la deteccion de rostros
print("calculando deteccion de rostros...")
net.setInput(blob)
detections = net.forward()

# itero en cada deteccion
for i in range(0, detections.shape[2]):
    #extraigo la probabilidad asociada a cada deteccion
    confidence = detections[0, 0, i, 2]
    # filtro las detecciones que tienen una confianza mayor a la minima preestablecida
    if confidence > args["confidence"]:
        #calculo las coordenadas (x,y) del cuadro delimitador del rostro
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int") #convierte las coordenadas del cuadro delimitador a enteros
        # me aseguro que las dimensiones del cuadro delimitador esten dentro de las dimensiones de la imagen
        (startX, startY) = (max(0, startX), max(0, startY)) 
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        #Lo siguiente es pasar nuestra region de interes (ROI) por el detector de barbijos
        # Extraigo el ROI y convierte de BGR a RGB
        face = image[startY:endY, startX:endX] #extraigo el rostro mediante slicing
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        #Redimensiono al tamano de entrada requirido por el modelo
        face = cv2.resize(face, (224, 224))
        #convierto a array
        face = img_to_array(face)
        # escalo los valores de pixeles a [-1,1]
        face = preprocess_input(face)
        # agrego una dimencion para el lote
        face = np.expand_dims(face, axis=0)
        # paso el rostro por el modelo para determina si tiene barbijo o no
        (mask, withoutMask) = model.predict(face)[0]

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Con_Barbijo" if mask > withoutMask else "Sin_Barbijo"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255) #asignamos color verde si tiene barbijo y rojo sino lo tiene
        # incluyo la probabilidad en la etiqueta
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # muestro la etiqueta y el cuadro delimitador
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
# muestro la imagen
def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure('as')
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()
plt_imshow("Output", image)

