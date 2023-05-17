# Bootstrap
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

df = sns.load_dataset('iris')
X = df[['sepal_width', 'sepal_length']].values.reshape((-1, 2))
y = df.species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
modelo = BaggingClassifier(estimator = knn, n_estimators = 50, max_samples = 0.3)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
accuracy_score(y_test, y_pred)
