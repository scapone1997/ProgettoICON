import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Carica il dataset
data = pd.read_csv('dataset.csv')

# Selezione delle variabili
features = ['Unit price', 'Quantity', 'Product line', 'Gender', 'Payment', 'Branch']

# Separare le variabili categoriali e numeriche
numerical_features = ['Unit price', 'Quantity']
categorical_features = ['Product line', 'Gender', 'Payment', 'Branch']

# Creare i trasformatori
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Creare il preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Applicare il preprocessing
X = preprocessor.fit_transform(data[features])

# Determinare il numero ottimale di cluster con il metodo del gomito
inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Grafico del metodo del gomito
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Numero di Cluster')
plt.ylabel('Inertia')
plt.title('Metodo del Gomito')
plt.show()

# Applicare K-means (supponiamo che 3 sia il numero ottimale)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Aggiungere i cluster al DataFrame originale
data['Cluster'] = clusters

# Esplorare le caratteristiche medie dei clienti in ciascun cluster
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)

# Ridurre le dimensioni per la visualizzazione (PCA)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Aggiungere le coordinate ridotte al DataFrame
data['Dimension1'] = X_reduced[:, 0]
data['Dimension2'] = X_reduced[:, 1]

# Visualizzazione dei cluster
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Dimension1', y='Dimension2', hue='Cluster', data=data, palette='viridis')
plt.title('Visualizzazione dei Cluster')
plt.show()
