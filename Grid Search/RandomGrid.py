import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

df = sns.load_dataset('tips')
X = df.tip.values.reshape((-1, 1))
y = df.total_bill

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeRegressor()
modelo.get_params()

params = {
    'criterion': ['absolute_error', 'squared_error'],
    'max_depth': [None, 5, 10]
}

grid_model = RandomizedSearchCV(modelo, params, cv = 5, n_iter = 2)
grid_model.fit(X_train, y_train)

mejor_modelo = grid_model.best_estimator_
y_pred = mejor_modelo.predict(X_test)
