import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generar el conjunto de datos simulado
X, y = make_blobs(n_samples=200, centers=3, random_state=42)

# Visualizar los puntos de datos
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

df = pd.DataFrame({'Feature 1': X[:, 0], 'Feature 2': X[:, 1], 'y': y})
X = df[['Feature 1', 'Feature 2']].values.reshape((-1, 2))
y = df.y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = KMeans(n_clusters=3)
k.fit(X_train, y_train)
y_pred = k.predict(X_test)

rdo = pd.DataFrame({'F1': X_test[:, 0],
                    'F2': X_test[:, 1],
                    'y': y_test,
                    'y_pred': y_pred
                    })

k.cluster_centers_
sns.scatterplot(data = rdo, x='F1', y = 'F2', hue = 'y_pred')
sns.scatterplot(x = k.cluster_centers_[:, 0], y = k.cluster_centers_[:, 1], label = 'Centroides')

k.inertia_
