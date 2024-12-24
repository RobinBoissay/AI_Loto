import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Charger une image d'exemple
img = cv2.imread("boule_25.jpg")

# Redimensionner pour une cohérence
img = cv2.resize(img, (64, 64))

# Créer un générateur d'augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotation aléatoire
    width_shift_range=0.2,  # Décalage horizontal
    height_shift_range=0.2,  # Décalage vertical
    brightness_range=[0.8, 1.2],  # Changement de luminosité
    zoom_range=0.2,  # Zoom aléatoire
    horizontal_flip=False,  # Ne pas retourner horizontalement (les nombres changeraient)
    fill_mode='nearest'
)

# Générer des images augmentées
img = img.reshape((1, 32, 32, 3))  # Reshape pour le générateur
save_dir = "Dataset/train/25/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Enregistrer les images générées
i = 0
for batch in datagen.flow(img, batch_size=1, save_to_dir=save_dir, save_prefix="boule_25", save_format="jpg"):
    i += 1
    if i >= 500:  # Générer 50 images augmentées
        break
