import cv2
import os

def split_image_into_balls(image_path, output_dir, rows=9, cols=10):
    """
    Divise une image contenant une grille de boules en plusieurs images individuelles.

    :param image_path: Chemin vers l'image source.
    :param output_dir: Dossier où les images individuelles seront sauvegardées.
    :param rows: Nombre de lignes dans la grille.
    :param cols: Nombre de colonnes dans la grille.
    """
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")

    # Dimensions de l'image
    height, width, _ = image.shape

    # Calcul des dimensions de chaque boule
    cell_height = height // rows
    cell_width = width // cols

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Découper l'image
    for row in range(rows):
        for col in range(cols):
            # Calculer les coordonnées de chaque cellule
            x_start = col * cell_width
            y_start = row * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height

            # Extraire la cellule
            ball = image[y_start:y_end, x_start:x_end]

            # Nommer et sauvegarder l'image de la boule
            ball_filename = os.path.join(output_dir, f"boule_{row * cols + col + 1}.jpg")
            cv2.imwrite(ball_filename, ball)

    print(f"Toutes les boules ont été extraites et sauvegardées dans : {output_dir}")

# Exemple d'utilisation
split_image_into_balls(
    image_path="imageBoules.png",  # Chemin vers votre image de la grille
    output_dir="boules",            # Dossier de sortie
    rows=9,                         # Nombre de lignes
    cols=10                         # Nombre de colonnes
)
