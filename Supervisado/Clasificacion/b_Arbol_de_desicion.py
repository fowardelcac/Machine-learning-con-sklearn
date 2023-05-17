import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score

df = sns.load_dataset('iris')
X = df.drop('species', axis=1)
y = df.species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(export_text(modelo, feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))
plt.figure(figsize=(15,8))
plot_tree(modelo, feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.show()

''' Creditos: 
Gutiérrez-García, J.O. [Código Máquina]. (2021, 18 de Octubre). Árboles de Decisión (decision trees) usando Entropía con Python [Video]. YouTube. [https://www.youtube.com/watch?v=z5rmY-LV7ME].

'''
