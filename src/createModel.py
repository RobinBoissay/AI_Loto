import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
from tensorflow.keras.optimizers import Adam

# Gérer la mémoire GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
        )
    except RuntimeError as e:
        print(e)

# Définir le chemin vers le dossier contenant les images
image_dir = "Dataset/train"

# Taille des images et nombre de classes
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 8  # Nombre de classes (changer en 90 pour le dataset complet)

# Charger les images et les étiquettes en niveaux de gris
def load_images_and_labels(image_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(image_dir), key=lambda x: int(x))  # Trier pour garantir un ordre constant
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(image_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Charger en niveaux de gris
            if image is None:
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels), class_names

# Charger les données
images, labels, class_names = load_images_and_labels(image_dir)

# Reshape les images pour avoir un canal (niveaux de gris)
images = images.reshape(-1, 64, 64, 1)

# Normaliser les pixels des images entre 0 et 1
images = images.astype('float32') / 255.0

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Créer le modèle
model = Sequential()

# Première couche de convolution
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))  # Un seul canal
model.add(MaxPooling2D(pool_size=(2, 2)))

# Deuxième couche de convolution
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Troisième couche de convolution
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplatissement des données pour la couche dense
model.add(Flatten())

# Couche dense
model.add(Dense(128, activation='relu'))

# Couche de sortie
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compilation du modèle
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Ajuster le nombre d'époques selon les besoins
    batch_size=32
)

# Sauvegarder le modèle
model.save("image_classification_model_gray.h5")

# Évaluer le modèle sur les données de validation
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Loss sur les données de validation : {val_loss}")
print(f"Précision sur les données de validation : {val_accuracy}")
