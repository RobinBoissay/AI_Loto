import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def generate_augmented_images(input_dir, output_dir, target_size=(64, 64), augmentations_per_image=100):
    """
    Applique des augmentations à toutes les images d'un dossier et sauvegarde les résultats.

    :param input_dir: Dossier contenant les images originales.
    :param output_dir: Dossier de sortie pour les images augmentées.
    :param target_size: Taille cible pour les images (par défaut 64x64).
    :param augmentations_per_image: Nombre d'images augmentées à générer par image d'entrée.
    """
    # Créer un générateur d'augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,            # Rotation réduite
        width_shift_range=0.1,        # Décalage limité
        height_shift_range=0.1,
        brightness_range=[0.5, 1.5],# Changement subtil de luminosité
        zoom_range=0.1,               # Zoom limité
        horizontal_flip=False,        # Pas de retournement horizontal
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
        for batch in datagen.flow(img, batch_size=1, save_prefix=f"boule_{boule_number}_aug", save_format="jpg"):
            # Clipping pour éviter les artefacts de couleurs
            augmented_image = np.clip(batch[0], 0, 255).astype('uint8')

            # Sauvegarde manuelle si nécessaire
            save_path = os.path.join(boule_output_dir, f"aug_{i}.jpg")
            cv2.imwrite(save_path, augmented_image)



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
