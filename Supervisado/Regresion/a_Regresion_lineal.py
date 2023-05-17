import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = sns.load_dataset("tips")

sns.scatterplot(x = df.total_bill, y = df.tip, data = df)

X = df.total_bill.values.reshape((-1, 1))
y = df.tip

regresion = LinearRegression()
modelo = regresion.fit(X = X, y = y)
modelo.intercept_, modelo.coef_[0]

predecir = [[18], [21], [7],  [15], [50]]
y_pred = modelo.predict(X = predecir)

import matplotlib.pyplot as plt

plt.scatter(predecir, modelo.predict(predecir), color = 'gray')
plt.plot(predecir, modelo.predict(predecir), color = 'red')
plt.scatter(df.total_bill, df.tip
