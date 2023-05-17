import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def metricas(y_test, y_pred):
  matriz = confusion_matrix(y_test, y_pred)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  return matriz, accuracy, precision, recall, f1

df = sns.load_dataset("titanic")
mapeo = {'male': 0,
       'female': 1}
df.sex = df.sex.map(mapeo)
X = df['sex'].values.reshape((-1, 1))
y = df.survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

matriz, accuracy, precision, recall, f1 = metricas(y_test, y_pred)

matriz_df = pd.DataFrame(matriz)
sns.heatmap(matriz_df, annot=True, cmap='YlGnBu')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.show()
'''
accuracy = predicciones correctas / total predicciones

Precision = Verd. positivo / (verd pos + falso pos)

recall = verd pos / (verd pos + falso neg)

f1 = 2/((1/precision) + (1/recall))
'''
print("Accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

vn, fn, fp, vp, = 88, 22, 17, 52
print("Accuracy:", ((vp + vn) / len(y_pred)) )
print("precision:", ((vp) / (vp + fp)))
print("recall:", ((vp) / (vp + fn)))
