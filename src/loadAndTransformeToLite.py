import tensorflow as tf

# Charger le modèle TensorFlow sauvegardé
model = tf.keras.models.load_model("image_classification_model.h5")

# Convertir en modèle TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Sauvegarder le modèle TFLite
with open("modele_boules.tflite", "wb") as f:
    f.write(tflite_model)

print("Le modèle a été converti et sauvegardé en TFLite.")