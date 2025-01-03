import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU détecté : {gpus}")
else:
    print("Aucun GPU détecté.")


