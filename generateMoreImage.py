import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_augmented_images(input_dir, output_dir, target_size=(32, 32), augmentations_per_image=500):
    """
    Applique des augmentations à toutes les images d'un dossier et sauvegarde les résultats.

    :param input_dir: Dossier contenant les images originales.
    :param output_dir: Dossier de sortie pour les images augmentées.
    :param target_size: Taille cible pour les images (par défaut 32x32).
    :param augmentations_per_image: Nombre d'images augmentées à générer par image d'entrée.
    """
    # Créer un générateur d'augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,  # Rotation aléatoire
        width_shift_range=0.2,  # Décalage horizontal
        height_shift_range=0.2,  # Décalage vertical
        brightness_range=[0.8, 1.2],  # Changement de luminosité
        zoom_range=0.2,  # Zoom aléatoire
        horizontal_flip=False,  # Ne pas retourner horizontalement
        fill_mode='nearest'
    )

    # Vérifier les dossiers
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir chaque image du dossier d'entrée
    for filename in os.listdir(input_dir):
        # Charger l'image
        file_path = os.path.join(input_dir, filename)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Impossible de lire l'image : {file_path}")
            continue

        # Redimensionner pour la cohérence
        img = cv2.resize(img, target_size)

        # Reshape pour le générateur
        img = img.reshape((1, target_size[0], target_size[1], 3))

        # Créer un dossier spécifique pour chaque boule
        boule_number = os.path.splitext(filename)[0].split('_')[-1]  # Extraire le numéro
        boule_output_dir = os.path.join(output_dir, boule_number)
        os.makedirs(boule_output_dir, exist_ok=True)

        # Générer des images augmentées
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=boule_output_dir, save_prefix=f"boule_{boule_number}", save_format="jpg"):
            i += 1
            if i >= augmentations_per_image:
                break

        print(f"Images augmentées générées pour {filename} dans {boule_output_dir}")

# Exemple d'utilisation
generate_augmented_images(
    input_dir="boules",       # Dossier contenant vos 90 images de boules
    output_dir="Dataset/train",  # Dossier de sortie pour les images augmentées
    target_size=(64, 64),     # Taille cible
    augmentations_per_image=100  # Nombre d'images générées par image
)
