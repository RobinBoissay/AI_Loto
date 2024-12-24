from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
# Charger le modèle entraîné
model = load_model("image_classification_model.h5")

# Charger une image de test
image_path = "Dataset/train/88/boule_88_0_180.jpg"
image_path = "boules/boule_2.jpg"

image = cv2.imread(image_path)

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




# Prédire
predictions = model.predict(image_input)
predicted_class = np.argmax(predictions)
predicted_label = predicted_class + 1

print(f"Classe prédite : {predicted_label}")

# # Faire la prédiction
# predictions = model.predict(image_input)

# # Trouver la classe prédite
# predicted_class = np.argmax(predictions)
confidence = np.max(predictions)
print(f"Classe prédite : {predicted_class + 1}")
print(f"Confiance : {confidence:.2f}")

# print(f"Forme de l'image d'entrée : {image_input.shape}")  # Devrait être (1, 64, 64, 3)
# print(f"Valeurs min/max des pixels : {image_input.min()} / {image_input.max()}")  # Devrait être entre 0 et 1

# class_names = sorted(os.listdir(image_path))
# with open("class_names.txt", "w") as f:
#     for class_name in class_names:
#         f.write(f"{class_name}\n")

