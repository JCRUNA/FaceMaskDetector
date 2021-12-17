# FaceMaskDetector
Deteccion de Barbijos con OpenCV, Keras y Tensorflow :smiley:

En este repositorio se muestra como entrenar un detector de barbijos pasando por 2 etapas:
- Cargar el modelo preentrenadado Mobilenet disponible en la libreria Keras y entrenarlo con el dataset de imagenes de personas con barbijos y personas sin barbijos. Guardar el modelo entrenado. 
- Usar el modelo HaarCascades disponible en la libreria OpenCV para poder detectar rostros en en dataset, obteniendo asi, las coordenadas del bounding box.
- Usar la imagen del rostro y pasarlo a traves de nuestro modelo para detectar si la persona tiene o no barbijo.

Pasos para seguir correctamente el proyecto:
- Ver el archivo  train-mask-detector.py
- Ver el archivo detect_mask_image.py
- Ver el archivo detect_mask_video.py



