# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator #para aumento de datos
from tensorflow.keras.applications import MobileNetV2 #modelo preentrenado a usar
from tensorflow.keras.layers import AveragePooling2D 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# creo el parser
ap = argparse.ArgumentParser()
#agrego argumento obligatorio de ruta al dataset
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
#agrego argumento opcional de ruta al grafico
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
#agrego argumento opcional de ruta al modelo
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

'''-dataset: es la ruta al dataset de barbijos
   -plot: la ruta a la imagen donde graficamos loss y accuracy
   - model: la ruta al modelo'''


#defino los hiperparametros a usar en nuestro modelo

INIT_LR = 1e-4 #taza de aprendizaje inicia. Luego durante el entrenamiento ira disminuyendo para 
#una mejor convergencia.
EPOCHS = 40 #cantidad de iteraciones sobre el conjunto de entrenamiento
BS = 32 #tamano de lote 

#graba la lista de imagenes en nuestro directorio , luego inicializo listas de data, labels, boundingbox
#
print("Cargando imagenes")
imagePaths = list(paths.list_images(args["dataset"])) #lista de rutas a las imagenes
data = []
labels = []

# itero sobre cada ruta de imagen
for imagePath in imagePaths:
	# extraigo label de la ruta de imagen
	label = imagePath.split(os.path.sep)[-2]
	# cargo imagen y redimensiono a 224x224
	image = load_img(imagePath, target_size=(224, 224))
    # convierto en array
	image = img_to_array(image)
    #preproceso la imagen con la funcion de keras para que los pixeles esten en el rango 0,1
	image = preprocess_input(image)
	# agrego la imagen a la lista data y la etiqueta a la lista labels
	data.append(image)
	labels.append(label)
# convierto las listas en array
data = np.array(data, dtype="float32")
labels = np.array(labels)

## Aun no finalizamos la preparacion de datos. Codificaremos nuestras etiquetas
# y los preparamos para el aumento de datos.

# inicializo instancia de LabelBinarizer y convierte las etiquetas a valores 0,1

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# divido el conjunto de entrenamiento y prueba en 80-20
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
# creo el ImageDatagenerator con parametros de rotacion, zoom, deslizamiento, etc para aplicar
# a las imagenes durante el entrenamiento. Esto crea nuevas imagenes a partir de las imagenes del
# conjunto de entrenamiento pero solo en memoria!! no las guarda!!
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#Preparamos Movilenet. Cargamos el modelo y nos aseguramos de no incluir la capas FC a traves del 
#parametro include_top=fale.
#Movilente recibe como input imagenes de 224x224x3 (3 canales: RGB)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
#Construimos la cabecera del modelo
#Para construir el modelo usamos la clase Model de Keras. Agregamos la salida del modelo Movilenet a la variable
#headModel y luego vamos agregando las capas a esta variable.
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel) 
headModel = Flatten(name='Flatten')(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#creo el modelo final ingresando como entrada la base y salida mi cabecera
model = Model(inputs=baseModel.input, outputs=headModel)

#itero por cada capa del modelo mobilenet y congelo los pesos de cada capa

for layer in baseModel.layers:
    layer.trainable = False


#Ya con el modelo armado podemos pasar a la etapa de compilacion
opt = Adam(learning_rate=INIT_LR,decay = INIT_LR/EPOCHS)
print("Compilando el modelo...")
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])
#La tasa de aprendizaje ira disminuyendo a medida que aumentan las epocas. Esto logra una mejor convergencia.
# la perdida usada es binary porque tenemos 2 clases. Si tuvieramos mas seria categorical. La metrica a medir es el
#Accuracy y el optimizador es adam.

#entreno la cabecera del modelo (headmodel) y lo asigno a la variable H(History)

H=model.fit(aug.flow(trainX,trainY,batch_size=BS),steps_per_epoch=len(trainX)//BS,
validation_data=(testX,testY), validation_steps=len(testX)//BS,epochs=EPOCHS)

#el primer parametro es para que la instancia de ImageDataGeneratos tomo las imagenes de entrenamiento y etiquetas en lotes de tamano BS
# steps_per_Epoch se define como la division entera entre cantidad de imagenes de entrenamiento y el tamno de lote.Esto daria la cantidad de ajuste de pesos por cada epoca.
# Tambien le pasamos las imagenes para validacion y el validation_steps

#Una vez entrenado el modelo debemos evaluarlo en el conjunto de prueba

print('Evaluando modelo ...')
predIdxs = model.predict(testX,batch_size=BS)

#para cada imagen de test debemos encontrar el indice de etiqueta correspondiente a la clase de mayor probabilidad
predIdxs = np.argmax(predIdxs,axis=1)

#muestro un reporte
print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

#guardo el modelo en el disco local
print("Guardando modelo de deteccion de barbijo")
model.save(args["model"], save_format="h5")

#El ultimo paso es trazar la loss y el accuracty para el conjunto de entrenamiento y validacion

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


