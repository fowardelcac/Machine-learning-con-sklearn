import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = sns.load_dataset('iris')
df.shape

scaler = StandardScaler()
data_t = scaler.fit_transform(df.drop('species', axis=1))

pca = PCA(n_components = 2)
pca.fit(data_t)
data_transf = pca.transform(data_t)
data_transf.shape

df_final = pd.DataFrame({'PCA1': data_transf[:, 0],
                         'PCA2': data_transf[:, 1] 
                         })

plt.figure(figsize=(3,2))
plt.scatter(data=df_final, x='PCA1', y='PCA2')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Gráfico de Dispersión PCA1 vs PCA2')
plt.show()

var_pca = pd.DataFrame({
    'columnas': ['PCA1', 'PCA2'],
    'var': pca.explained_variance_ratio_
})
sns.barplot(data = var_pca, x = 'columnas', y = 'var')
