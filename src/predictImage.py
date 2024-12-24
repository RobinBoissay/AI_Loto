import tensorflow as tf
import numpy as np
from PIL import Image

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path="modele_boules.tflite")
interpreter.allocate_tensors()

# Obtenir les informations sur les entrées et sorties
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input details", input_details)

print("output details", output_details)
image = Image.open("../boules/boule_53.jpg").convert("RGB")  # Convertir en RGB pour enlever le canal alpha
image = image.resize((64, 64))  # Redimensionner l'image
image_array = np.array(image)
noise_factor = 0.5
image_noisy = np.array(image) + noise_factor * np.random.randn(*image_array.shape)
image_noisy = np.clip(image_noisy, 0., 255.)
input_data = np.array(image_noisy) / 255.0  # Normaliser entre 0 et 1
input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # Ajouter la dimension batch


# Vérifier les dimensions de l'entrée
print(f"Forme de l'entrée préparée : {input_data.shape}")  # Devrait être [1, 32, 32, 3]

# Effectuer une prédiction
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
# Afficher la classe prédite
predicted_class = np.argmax(output_data)
print(f"Classe prédite : {predicted_class + 1}")
print(predicted_class)

top_5 = np.argsort(output_data[0])[-5:][::-1]
print(f"Top 5 classes prédites : {top_5 + 1}")
print(f"Probabilités correspondantes : {output_data[0][top_5]}")