from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Charger le modèle entraîné
model = load_model("image_classification_model.h5")

# Charger une image de test
image_path = "Dataset/train/31/frame_00025.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Vérifier si l'image est correctement chargée
if image is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarisation (seuil adaptatif pour distinguer le texte)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optionnel : Inverser les couleurs si nécessaire
    inverted = cv2.bitwise_not(binary)

    # Étendre l'image pour avoir 3 canaux (nécessaire pour le modèle)
    image_three_channels = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

    # Redimensionner l'image
    image_resized = cv2.resize(image_three_channels, (64, 64))

    # Normaliser les pixels
    image_normalized = image_resized.astype('float32') / 255.0

    # Ajouter une dimension pour correspondre au format attendu par le modèle (lot de données)
    image_input = np.expand_dims(image_normalized, axis=0)

    # Prédire
    predictions = model.predict(image_input)
    predicted_class = np.argmax(predictions)
    predicted_label = predicted_class + 1  # Si les labels commencent à 1

    confidence = np.max(predictions)
    print(f"Classe prédite : {predicted_label}")
    print(f"Confiance : {confidence:.2f}")
