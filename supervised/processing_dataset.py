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


def trasforma_in_dummies(dataset, colonne_categoriali):
    """
    Trasforma le colonne categoriali specificate in variabili dummy.

    Parameters:
    - dataset (pd.DataFrame): Il dataframe originale con le colonne da trasformare.
    - colonne_categoriali (list): Lista dei nomi delle colonne categoriali da trasformare in dummy.

    Returns:
    - pd.DataFrame: Il dataframe con le colonne categoriali trasformate in variabili dummy.
    """
    # Verifica che le colonne categoriali esistano nel dataset
    colonne_presenti = [col for col in colonne_categoriali if col in dataset.columns]

    # Se alcune colonne non sono presenti, stampa un avviso
    colonne_assenti = set(colonne_categoriali) - set(colonne_presenti)
    if colonne_assenti:
        print(f"Attenzione: le seguenti colonne non sono presenti nel dataset: {colonne_assenti}")

    # Trasforma le colonne categoriali in dummy e unisce con le altre colonne
    dataset_dummies = pd.get_dummies(dataset, columns=colonne_presenti, drop_first=True)

    # Rinomina le colonne dummy per riflettere il nome della colonna originale
    for col in colonne_presenti:
        dummy_columns = [c for c in dataset_dummies.columns if c.startswith(col + '_')]
        for dummy_col in dummy_columns:
            # Rinominazione: 'colonna_originale_valore' -> 'colonna_originale'
            new_col_name = col
            dataset_dummies = dataset_dummies.rename(columns={dummy_col: new_col_name})

    return dataset_dummies

def separa_variabili(dataset):
    # Variabile target Y (G3)
    Y = dataset['G3']

    # Variabili indipendenti X (tutte tranne G3)
    X = dataset.drop('G3', axis=1)

    return X, Y
