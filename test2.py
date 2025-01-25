import cv2
import pytesseract

# Configuration de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_number(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des cercles (boules) avec HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)
    
    if circles is None:
        print("Aucun cercle détecté.")
        return
    
    # Dessiner les cercles détectés pour vérifier
    circles = circles[0, :].astype("int")
    for (x, y, r) in circles:
        print(f"Cercle détecté à ({x}, {y}) avec rayon {r}")
        
        # Vérifier les limites de la ROI
        height, width = gray.shape
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(width, x + r), min(height, y + r)

        print(f"ROI de ({x1}, {y1}) à ({x2}, {y2})")

        roi = gray[y1:y2, x1:x2]
        
        # Vérifier si ROI est vide
        if roi.size == 0:
            print("Région d'intérêt vide ou invalide.")
            continue

        # Appliquer un seuil adaptatif
        roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

        # Reconnaître le texte avec Tesseract
        config = "--psm 8 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(roi_thresh, config=config)
        text = text.strip()

        if text.isdigit():
            print(f"Numéro détecté : {text}")
        else:
            print("Aucun numéro valide détecté.")

    # Afficher l'image avec les résultats
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Appeler la fonction avec une image spécifique
detect_number("test/imageTest15.jpg")
