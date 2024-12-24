import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path="modele_boules.tflite")
interpreter.allocate_tensors()

# Obtenir les informations sur les entrées et sorties
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input details", input_details)

print("output details", output_details)

image = cv2.imread("boules/boule_10.jpg")

# Vérifier si l'image est correctement chargée
if image is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Redimensionner l'image
    image_resized = cv2.resize(image, (64, 64))
    
    # Normaliser les pixels
    image_normalized = image_resized.astype('float32') / 255.0

    # Ajouter une dimension pour correspondre au format attendu par le modèle (lot de données)
    image_input = np.expand_dims(image_normalized, axis=0)
    
# Normaliser les pixels
image_normalized = image_resized.astype('float32') / 255.0

# Ajouter une dimension pour correspondre au format attendu par le modèle (lot de données)
image_input = np.expand_dims(image_normalized, axis=0)

# Vérifier les dimensions de l'entrée
print(f"Forme de l'entrée préparée : {image_input.shape}")  # Devrait être [1, 32, 32, 3]

# Effectuer une prédiction
interpreter.set_tensor(input_details[0]['index'], image_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
# Afficher la classe prédite
predicted_class = np.argmax(output_data)
print(f"Classe prédite : {predicted_class + 1}")

top_5 = np.argsort(output_data[0])[-5:][::-1]
print(f"Top 5 classes prédites : {top_5 + 1}")
print(f"Probabilités correspondantes : {output_data[0][top_5]}")