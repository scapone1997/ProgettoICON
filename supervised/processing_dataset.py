import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def prepara_dataset(nome_file_csv):
    # Carica il dataset
    dataset = pd.read_csv(nome_file_csv)

    # Seleziona le colonne categoriali che vuoi codificare (escludendo la variabile target 'G3')
    colonne_categoriali = dataset[['famsize', 'address']]

    # Applica OneHotEncoding alle colonne categoriali
    encoder = OneHotEncoder(drop='first')
    encoded_categoriali = encoder.fit_transform(colonne_categoriali).toarray()

    # Crea un DataFrame dalle colonne codificate
    colonne_encoded_df = pd.DataFrame(encoded_categoriali, columns=encoder.get_feature_names_out(colonne_categoriali.columns))

    # Unisci il dataset con le colonne codificate
    dataset_finale = pd.concat([dataset, colonne_encoded_df], axis=1)

    # Rimuovi le colonne originali categoriali (ma non la variabile target 'G3')
    dataset_finale.drop(['famsize', 'address'], axis=1, inplace=True)

    # Restituisci il dataset finale
    return dataset_finale

def separa_variabili(dataset):
    # Variabile target Y (G3)
    Y = dataset['G3']

    # Variabili indipendenti X (tutte tranne G3)
    X = dataset.drop('G3', axis=1)

    return X, Y
