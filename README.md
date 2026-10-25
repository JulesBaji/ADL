# Détection de photos recapturées

Projet d'apprentissage profond du master Informatique ISICG (Université de Limoges, 2024), réalisé en groupe.

Une photo recapturée est la photo d'un écran ou d'une impression. Le programme dit si une image est une photo directe ou une recapture.

## Méthode

1. **Descripteurs** (`Recapture/features/`) : descripteurs RID-SPIC d'après un article de recherche (erreurs de prédiction, différences secondes, descripteurs LTC), soit un vecteur de 324 valeurs par image.
2. **Classifieur** (`Recapture/training.py`) : réseau de neurones dense (324-128-64-1) en Keras, avec réduction du taux d'apprentissage et sauvegarde du meilleur modèle.
3. **Évaluation** (`Recapture/tests.py`) : jeu de test séparé, avec recadrages aléatoires. **91 % de précision** sur le jeu de test.

Des photos capturées et recapturées à la main servent de tests supplémentaires (`Capture_manuelle_tests/`, `Recapture_manuelle_tests/`).

## Utilisation

```bash
cd Recapture
python3 training.py      # entraînement (ou chargement du modèle de modele/)
python3 prediction.py    # prédiction sur un dossier d'images
python3 tests.py         # évaluation
```

Nécessite Python 3, TensorFlow/Keras, scikit-learn, scikit-image et NumPy. Les descripteurs et les poids entraînés sont fournis dans `modele/`.

## Technologies

Python, TensorFlow/Keras, scikit-learn, scikit-image, NumPy.
