import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_squared_error

def metricas(y_test, y_pred):
  error_max = max_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)
  rss = mse * len(y_pred)
  rmse = mean_squared_error(y_test, y_pred, squared = False)
  r2 = r2_score(y_test, y_pred)
  return error_max, mae, mse, rss, rmse, r2

df = sns.load_dataset("tips")

X = df.total_bill.values.reshape((-1, 1))
y = df.tip

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regresion = LinearRegression()
modelo = regresion.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
error_max, mae, mse, rss, rmse, r2 = metricas(y_test, y_pred)

print("Error max: ", error_max)
print("MAE:", mae)
print("MSE:", mse)
print("RSS:", rss)
print("RMSE:", rmse)
print("R^2:", r2)
