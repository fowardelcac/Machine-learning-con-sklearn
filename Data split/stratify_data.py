import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

n_samples = 1000  # Número total de muestras
n_features = 2  
imbalance_ratio = 0.4  

# Crear el dataset sintético desbalanceado
X, y = make_classification(
    n_samples=n_samples, #número total de muestras 
    n_features=n_features, # Número de características
    n_informative=int(n_features * 0.8), # Representa el número de características que son informativas y tienen un impacto directo en la variable de destino (clases). 80% de n_features
    n_redundant=int(n_features * 0.2), #número de características que son redundantes y se generan como combinaciones lineales de características informativas
    n_clusters_per_class=1, #Especifica el número de grupos/clusters por clase. En este caso, se establece en 1, lo que significa que las muestras de cada clase se agrupan en un solo cluster.
    weights=[imbalance_ratio], # Razón de desbalanceo (1 indica un balanceo perfecto)
    flip_y=0.01, #Controla el ruido en las etiquetas de clase 1% de ruido en las etiquetas.
    random_state=42
)

X.shape
df = pd.DataFrame({
    'X1': X[:, 0],
    'X2': X[:, 1],
    'Y': y
})

df.Y.value_counts().plot(kind='barh')

X = df[['X1', 'X2']].values.reshape((-1, 2))
y = df.Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Imprimir la distribución de clases en los conjuntos de entrenamiento y prueba
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)
class_counts_train = dict(zip(unique_train, counts_train))
class_counts_test = dict(zip(unique_test, counts_test))
print("Distribución de clases en el conjunto de entrenamiento:")
print(f'La Clase 0 reprsenta el: {(class_counts_train[0] / (class_counts_train[0] + class_counts_train[1])) * 100}%', 'La muestra es:', class_counts_train)
print("Distribución de clases en el conjunto de prueba:")
print(f'La Clase 0 reprsenta el: {(class_counts_test[0] / (class_counts_test[0] + class_counts_test[1])) * 100}%', 'La muestra es:', class_counts_test)

pd.Series(y_train).value_counts().plot(kind='barh')
pd.Series(y_test).value_counts().plot(kind='barh')
