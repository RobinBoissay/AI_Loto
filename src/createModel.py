import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2

# Définir le chemin vers le dossier contenant les images
image_dir = "Dataset/train"

# Taille des images et nombre de classes
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 90 # Nombre de classes à deviner, dans notre cas 90 parce que 90 boules.

# Charger les images et les étiquettes
def load_images_and_labels(image_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(image_dir), key=lambda x: int(x)) # Trier pour garantir un ordre constant
    for label, class_name in enumerate(class_names):
        print(label)
        class_path = os.path.join(image_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels), class_names


# Charger les données
images, labels, className = load_images_and_labels(image_dir)



# Normaliser les pixels des images entre 0 et 1
images = images.astype('float32') / 255.0

# Convertir les étiquettes en format one-hot
labels = to_categorical(labels, NUM_CLASSES)

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Créer le modèle
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Vous pouvez ajuster le nombre d'époques
    batch_size=32
)

# Sauvegarder le modèle
model.save("image_classification_model.h5")

# Évaluer le modèle sur les données de validation
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Loss sur les données de validation : {val_loss}")
print(f"Précision sur les données de validation : {val_accuracy}")
