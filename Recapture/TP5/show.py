import matplotlib.pyplot as plt
import numpy as np

def show(x, y): 
    # Définir le nombre de lignes et de colonnes pour la grille d'images
    num_rows = 4
    num_cols = 10

    # Afficher les images pour chaque label
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    for i in range(num_rows):
        for j in range(num_cols):
            label = i * num_cols + j  # Calculer le label correspondant à la position dans la grille
            image_index = np.where(y == label)[0][0]  # Obtenir l'index de la première image avec ce label
            axes[i, j].imshow(x[image_index], cmap='gray')  # Afficher l'image en niveaux de gris
            axes[i, j].set_title(f"Label {label}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()