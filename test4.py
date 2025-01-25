import cv2
import os

def extract_frames(video_path):
    """
    Extrait les frames d'une vidéo, applique un traitement d'image et les sauvegarde dans un dossier.
    :param video_path: Chemin de la vidéo (MP4, AVI, etc.)
    """
    # Vérifier si la vidéo existe
    if not os.path.exists(video_path):
        print(f"Erreur : La vidéo '{video_path}' est introuvable.")
        return
    
    # Obtenir le nom de la vidéo (sans extension)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Créer un dossier pour stocker les frames
    output_dir = f"{video_name}_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Les frames seront enregistrées dans : {output_dir}/")
    
    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Initialiser le compteur de frames
    frame_number = 0

    while True:
        ret, frame = cap.read()  # Lire une frame
        if not ret:  # Fin de la vidéo
            break

        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Appliquer un flou pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binarisation (seuil adaptatif pour distinguer le texte)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Optionnel : Inverser les couleurs si nécessaire (texte sombre sur fond clair attendu par Tesseract)
        inverted = cv2.bitwise_not(binary)

        # Nom du fichier de la frame traitée
        frame_filename = os.path.join(output_dir, f"frame_{frame_number:05d}.jpg")

        # Sauvegarder la frame traitée comme image
        cv2.imwrite(frame_filename, inverted)
        print(f"Frame {frame_number} sauvegardée : {frame_filename}")
        
        frame_number += 1

    # Libérer les ressources
    cap.release()
    print("Extraction terminée.")

# Exemple d'utilisation
video_path = "video/boule9.mp4"  # Remplace par le chemin de ta vidéo
extract_frames(video_path)
