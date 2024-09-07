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

# Applicare K-means con 6 cluster
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(X)

# Aggiungere i cluster al DataFrame originale
data['Cluster'] = clusters

# Selezionare solo colonne numeriche per l'aggregazione
numeric_data = data[['Unit price', 'Quantity', 'Cluster']]

# Esplorare le caratteristiche medie dei clienti in ciascun cluster
cluster_summary = numeric_data.groupby('Cluster').mean()
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
plt.title('Visualizzazione dei Cluster con K-means (6 Cluster)')
plt.show()


# Interpretazione dei Risultati
# Cluster 0:
#
# Unit Price: 78.80
# Quantity: 3.64
# Interpretazione: I clienti in questo cluster tendono ad acquistare prodotti a un prezzo unitario medio più alto (78.80$) e acquistano una quantità media di circa 3.64 unità per transazione.
# Cluster 1:
#
# Unit Price: 36.07
# Quantity: 8.17
# Interpretazione: I clienti in questo cluster acquistano prodotti a un prezzo unitario medio più basso (36.07$) ma acquistano una quantità media di circa 8.17 unità per transazione. Questo potrebbe indicare che i clienti di questo cluster acquistano più prodotti, ma a un prezzo inferiore.
# Cluster 2:
#
# Unit Price: 81.63
# Quantity: 8.44
# Interpretazione: I clienti in questo cluster acquistano prodotti a un prezzo unitario medio molto alto (81.63$) e acquistano una quantità media di circa 8.44 unità per transazione. Questo potrebbe indicare clienti che spendono molto per transazione e acquistano una quantità significativa di prodotti costosi.
# Cluster 3:
#
# Unit Price: 71.94
# Quantity: 2.87
# Interpretazione: I clienti in questo cluster tendono ad acquistare prodotti a un prezzo unitario relativamente alto (71.94$) ma acquistano una quantità media inferiore di circa 2.87 unità per transazione. Questo potrebbe indicare acquisti di prodotti costosi ma in quantità limitata.
# Cluster 4:
#
# Unit Price: 33.75
# Quantity: 7.33
# Interpretazione: I clienti in questo cluster acquistano prodotti a un prezzo unitario medio relativamente basso (33.75$) e acquistano una quantità media di circa 7.33 unità per transazione. Questo suggerisce che i clienti di questo cluster fanno acquisti relativamente più economici e in quantità maggiore rispetto ad altri cluster.
# Cluster 5:
#
# Unit Price: 29.54
# Quantity: 2.84
# Interpretazione: I clienti in questo cluster tendono ad acquistare i prodotti a un prezzo unitario molto basso (29.54$) e acquistano una quantità media di circa 2.84 unità per transazione. Questo potrebbe indicare acquisti a basso costo e in quantità limitata.
