import numpy as np

# Création de deux tableaux booléens de même dimension
array1 = np.array([True, False, True])
array2 = np.array([False, True, False])

# Opération de somme entre les deux tableaux
somme = array1 + array2
print("Somme des tableaux :", somme)

# Opération de "ou" logique entre les deux tableaux
ou_logique = np.logical_or(array1, array2)
print("Opération de 'ou' logique :", ou_logique)
