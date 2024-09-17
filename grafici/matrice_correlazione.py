import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def visualizza_matrice_correlazione(nome_file_csv):
    # Carica il dataset
    dataset = pd.read_csv(nome_file_csv)

    # Separa colonne numeriche e categoriali
    colonne_numeriche = dataset.select_dtypes(include=['number'])
    colonne_non_numeriche = dataset.select_dtypes(include=['object'])

    # Stampa le colonne numeriche e categoriali
    print("Colonne numeriche:")
    print(colonne_numeriche.head())  # Mostra le prime 5 righe delle colonne numeriche

    print("\nColonne categoriali:")
    print(colonne_non_numeriche.head())  # Mostra le prime 5 righe delle colonne categoriali

    # Codifica le colonne categoriali e le aggiunge al dataset
    dataset_completo = _codifica_colonne_categoriali(colonne_numeriche, colonne_non_numeriche)

    # Visualizza la matrice di correlazione
    _plot_heatmap_correlazione(dataset_completo.corr(), 'Matrice di Correlazione')


def _codifica_colonne_categoriali(colonne_numeriche, colonne_non_numeriche):
    """Codifica le colonne categoriali usando OneHotEncoder e le concatena alle colonne numeriche."""
    encoder = OneHotEncoder(drop='first')
    colonne_categoriali_encoded = encoder.fit_transform(colonne_non_numeriche).toarray()
    colonne_categoriali_df = pd.DataFrame(colonne_categoriali_encoded,
                                          columns=encoder.get_feature_names_out(colonne_non_numeriche.columns))
    return pd.concat([colonne_numeriche, colonne_categoriali_df], axis=1)


def _plot_heatmap_correlazione(matrice_correlazione, titolo):
    """Crea una heatmap per visualizzare la matrice di correlazione."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrice_correlazione, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title(titolo)
    plt.show()


# Esempio di utilizzo
visualizza_matrice_correlazione('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student_combined.csv')
