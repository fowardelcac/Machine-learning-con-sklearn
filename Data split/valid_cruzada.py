import pandas as pd
import seaborn as sns

from sklearn.model_selection import KFold

df = sns.load_dataset('iris')
X = df.sepal_length.values.reshape((-1, 1))
y = df.species 

kfold = KFold(n_splits=2, shuffle=True, random_state = 5)

for train_i, test_i  in kfold.split(X):
  #print(train_i, test_i)
  #print("*" * 100)
  X_train, X_test = X[train_i], X[test_i]
  y_train, y_test = y[train_i], y[test_i]
  # Aca  se realiza la implementacion del modelo
