import cv2
import os

def extract_frames(video_path):
    """
    Extrait les frames d'une vidéo et les sauvegarde dans un dossier.
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

        # Nom du fichier de la frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_number:05d}.jpg")
        
        # Sauvegarder la frame comme image
        cv2.imwrite(frame_filename, frame)
        print(f"Frame {frame_number} sauvegardée : {frame_filename}")
        
        frame_number += 1

    # Libérer les ressources
    cap.release()
    print("Extraction terminée.")

# Exemple d'utilisation
video_path = "video/boule85.mp4"  # Remplace par le chemin de ta vidéo
extract_frames(video_path)
