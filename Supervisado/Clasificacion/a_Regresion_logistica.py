import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
modelo.score(X_test, y_test)
