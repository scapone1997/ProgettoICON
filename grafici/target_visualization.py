from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def visualizza_target(nome_file: str):
    # Leggi il dataset dal file CSV
    dataset = pd.read_csv(nome_file)

    # Creare la prima figura con una griglia di 4 righe e 4 colonne (per i primi 16 grafici)
    fig1, griglia_one = plt.subplots(1, 3, figsize=(20, 16))  # Layout 4x4

    # PRIMA FIGURA (16 grafici)
    # Grafico per la distribuzione di 'school'
    sns.countplot(x='G1', data=dataset, ax=griglia_one[0])
    griglia_one[0].set_title('Distribuzione G1')
    griglia_one[0].set_xlabel('G1')
    griglia_one[0].set_ylabel('Conteggio')

    # Grafico per la distribuzione di 'age'
    sns.countplot(x='G1', data=dataset, ax=griglia_one[1])
    griglia_one[1].set_title('Distribuzione G2')
    griglia_one[1].set_xlabel('G2')
    griglia_one[1].set_ylabel('Conteggio')

    # Grafico per la distribuzione di 'sex'
    sns.countplot(x='G3', data=dataset, ax=griglia_one[2])
    griglia_one[2].set_title('Distribuzione G3')
    griglia_one[2].set_xlabel('G3')
    griglia_one[2].set_ylabel('Conteggio')


# Chiama il metodo per visualizzare i dati
#visualizza_target(r"C:\Users\simone.capone\PycharmProjects\ProgettoICON\dataset\student-mat.csv")
visualizza_target(r"C:\Users\simone.capone\PycharmProjects\ProgettoICON\dataset\student-por.csv")

plt.show()