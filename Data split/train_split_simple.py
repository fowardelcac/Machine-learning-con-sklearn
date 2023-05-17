import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

df = sns.load_dataset("tips")
df.head()

X = df.drop('total_bill', axis = 1)
y = df.total_bill

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
