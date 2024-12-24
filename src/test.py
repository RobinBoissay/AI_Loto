import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# Configurer le générateur pour lire les données d'entraînement
datagen = ImageDataGenerator(validation_split=0.2)  # Validation à 20%

train_gen = datagen.flow_from_directory(
    'Dataset/train',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Obtenir un batch d'images et de labels
images, labels = next(train_gen)

# Afficher quelques exemples
plt.figure()
for i in range(9):  # Montrer les 9 premières images
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(f"Classe : {np.argmax(labels[i]) + 1}")
    plt.axis('off')
plt.show()