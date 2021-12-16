# importo las librerias necesarias
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

#el script es similar al anteriro solo que ahora procesaremos cada frame del video
# para mayor conveniencia crearemos una funcion
def detect_and_predict_mask(frame, faceNet, maskNet):
    # asigno a variables las dimensiones ancho y alto del frame y creo el blob donde redimensiono y normalizo el frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    # paso el blob por el detecto de caras
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # inicializo lista de rostros, lista de las coordenas del cuadro delimitador y lista de predicciones de nuestro detecto de barbijos
    faces = []
    locs = []
    preds = []

#Esta funcion acepta 3 parametros, frame que es el frame de entrada, facenet el modelo de deteccion de rostros y
# masknet que es el modelo de deteccion de barbijos
# Luego inicializa listas, que retorna los rostros , (las ubicaciones de las caras), y preds (la lista de predicciones de barbijo / no barbijo).

#luego iteramos en cada deteccion obtenida

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extraigo las probabilidad asociadas a cada deteccion
        confidence = detections[0, 0, i, 2]
        # filtro las que tienen una confianza mayor a la deseada
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # me asegura que las dimensiones del cuadro delimitador caigan dentro de las dimensiones del frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


#ahora pasamos el ROI de los rostros a 2 de nuestras listas
            face = frame[startY:endY, startX:endX]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
#Ahora podemos pasar nuestra roi por el modelo de deteccion de barbijos

    # me aseguro de predecir siempre que se haya detectado un rostro
    if len(faces) > 0:
        #para mejorar la prediccion y hacerla mas rapida en en metodo  predict
        # le pasamos un lote de 32 frames.
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    # retorna una tupla de las coordenas del rostro y su probabilidads de deteccion de barbijo o no
    return (locs, preds)

#construyo el argparser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
  default="face-mask-detector",
  help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
  default="mask_detector.model",
  help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
  help="minimum probability to filter weak detections")
ap.add_argument("-in","--input",default="video.mp4")
ap.add_argument("-o","--output",default="output.avi")
args = vars(ap.parse_args())

# nuestra linea de argumentos incluye:
# --face : la ruta al directorio face-mask-detector
# --model : la ruta al modelo entrenado de deteccion de barbijos
# --confidence : la minima confianza requerida para que un la imagen considere que hay un rostro

# load our serialized face detector model from disk
print("Cargando detector de rostros...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
print("Cargando detector de barbijos")
maskNet = load_model(args["model"])
# initialize the video stream and allow the camera sensor to warm up
print("Iniciando Video Streaming..")
# inicio carga del video
vs =  cv2.VideoCapture(args["input"])
writer = None

# iteramos en cada frame del video
while True:
    #leo el frame y lo dimension a un ancho de 400 pixeles
    frame = vs.read()[1]
    if frame is None:
        break


    frame = imutils.resize(frame, width=400)
    # detector los rostros e identifico si tienen o no barbijo
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # itero en cada cuadro delimitados y prediccion
    for (box, pred) in zip(locs, preds):
        # desagrego los componenes del box y de la prediccion en tuplas
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        # determino la clase y el color a dibujar segun el caso
        label = "Con_barbijo" if mask > withoutMask else "Sin_Barbijo"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # incluyo la probabilidad en la etiqueta
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # agrego la etiqueta y el box en el frame actual
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
      (frame.shape[1], frame.shape[0]), True)
  
  # if the writer is not None, write the frame to disk
    if writer is not None:
        writer.write(frame)

# do a bit of cleanup
vs.release()

# check to see if the video writer point needs to be released
if writer is not None:
  writer.release()