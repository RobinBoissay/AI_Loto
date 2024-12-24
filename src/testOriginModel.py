from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Charger le modèle entraîné
model = load_model("image_classification_model.h5")

# Charger une image de test
image_path = "boules/boule_43.jpg"
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


# Faire la prédiction
predictions = model.predict(image_input)

# Trouver la classe prédite
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)
print(f"Classe prédite : {predicted_class}")
print(f"Confiance : {confidence:.2f}")


