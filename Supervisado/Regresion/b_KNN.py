import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df = sns.load_dataset('iris')
X = df['sepal_width'].values.reshape((-1, 1))
y = df.sepal_length
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = KNeighborsRegressor()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
