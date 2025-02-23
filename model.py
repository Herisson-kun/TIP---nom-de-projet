# -*- coding: utf-8 -*-
"""TIP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jAuemXXA21B7XqGNZ6Ea9fv7hrLoDCsx
"""

# Chemins vers les répertoires de dataset
base_dir = 'Normalized_Dataset'
train_dir = f"{base_dir}/Training"
test_dir = f"{base_dir}/Test"
val_dir = f"{base_dir}/Validation"

import tensorflow as tf
import matplotlib.pyplot as plt

# Paramètres de chargement
batch_size = 32
img_height = 256  # hauteur d'image que vous souhaitez redimensionner
img_width = 256   # largeur d'image que vous souhaitez redimensionner

# Charger les ensembles d'entraînement, de validation et de test
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',           # Infère les labels à partir des sous-dossiers
    label_mode='categorical',            # Encode les labels sous forme de one-hot vecteur classe 2 : (0, 1, 0)
    batch_size=batch_size,
    image_size=(img_height, img_width)
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width)
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width)
)

class_names = train_ds.class_names
"""
# Prendre un seul batch du dataset
for images, labels in train_ds.take(1):  # train_ds est votre dataset d'entraînement

    # Afficher chaque image et son label
    plt.figure(figsize=(15, 15))
    for i in range(batch_size):
        ax = plt.subplot(6, 6, i + 1)  # Créer une grille pour afficher les images (ajustez si nécessaire)
        plt.imshow(images[i].numpy().astype("uint8"))  # Convertir en uint8 pour affichage
        label = labels[i].numpy()

        # Vérifier si le label est un one-hot vector, sinon l'afficher directement
        if len(label) > 1:
            label = tf.argmax(label).numpy()  # Récupérer l'indice du label si one-hot
        plt.title(f"Label: {class_names[label]}")
        plt.axis("off")  # Désactiver les axes pour une meilleure lisibilité

    plt.show()"""

normalization_layer = tf.keras.layers.Rescaling(1./255)

# Appliquer la normalisation
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

"""Architecture CNN"""

from tensorflow.keras import layers, models

# Définir le modèle CNN
def create_cnn_model(input_shape=(256, 256, 3), num_classes=3):
    model = models.Sequential()

    # Première couche de convolution + max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Deuxième couche de convolution + max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Troisieme couche de convolution + max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Quatrieme couche de convolution + max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Cinquieme couche de convolution + max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Sixieme couche de convolution + max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Septieme couche de convolution + max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Couches denses (Fully connected)
    model.add(layers.Flatten())  # Aplatir l'entrée pour la couche dense
    model.add(layers.Dense(32, activation='relu'))  # Couche dense avec ReLU
    model.add(layers.Dropout(0.5))  # Dropout pour éviter le surapprentissage

    # Couche de sortie
    model.add(layers.Dense(num_classes, activation='softmax'))  # 3 classes avec activation softmax

    # Compiler le modèle
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Créer le modèle
model = create_cnn_model(input_shape=(256, 256, 3), num_classes=3)

# Afficher le résumé du modèle
model.summary()

# Entraînement du modèle avec une barre de progression
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    verbose=1  # Affiche une barre de progression et les informations de perte et d'exactitude
)

# Évaluation du modèle sur le dataset de test
test_loss, test_accuracy = model.evaluate(
    test_ds,  # Dataset de test
    steps=int(test_ds.cardinality()),  # Nombre de batchs dans le dataset de test
    verbose=1  # Afficher la progression de l'évaluation
)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Si vous avez un modèle entraîné, et que vous voulez générer la matrice de confusion sur test_ds
all_true_labels = []
all_pred_labels = []

# Parcourir tout le dataset de test
for images, labels in test_ds:
    # Faire des prédictions sur le batch d'images
    predictions = model.predict(images)
    # Obtenir les indices des classes prédites (l'index du maximum dans les prédictions)
    predicted_labels = np.argmax(predictions, axis=1)

    # Obtenir les indices des classes réelles (les labels)
    true_labels = np.argmax(labels, axis=1)

    # Ajouter les labels réels et prédits aux listes
    all_true_labels.extend(true_labels)
    all_pred_labels.extend(predicted_labels)

# Générer la matrice de confusion
cm = confusion_matrix(all_true_labels, all_pred_labels)

# Afficher la matrice de confusion
class_names = ['early_blight', 'healthy', 'late_blight']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vrais Labels')
plt.show()

# Récupérer l'historique d'entraînement
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

# Créer la figure avec deux sous-graphiques
plt.figure(figsize=(12, 5))

# Premier sous-graphique pour l'accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b', label='Accuracy entrainement')
plt.plot(epochs_range, val_acc, 'r', label='Accuracy validation')
plt.title('Évolution de l\'accuracy', fontsize=12)
plt.xlabel('Époques', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=9)

# Deuxième sous-graphique pour la loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b', label='Loss entrainement')
plt.plot(epochs_range, val_loss, 'r', label='Loss validation')
plt.title('Évolution de la loss', fontsize=12)
plt.xlabel('Époques', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=9)

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Afficher les graphiques
plt.show()

# Afficher les métriques finales
print("Métriques finales après", len(epochs_range), "époques:")
print(f"Accuracy entrainement : {acc[-1]:.4f}")
print(f"Accuracy validation  : {val_acc[-1]:.4f}")
print(f"Loss entrainement    : {loss[-1]:.4f}")
print(f"Loss validation     : {val_loss[-1]:.4f}")