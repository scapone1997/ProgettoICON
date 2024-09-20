import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualizza_dati_da_csv(nome_file: str, output_dir: str):
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Leggi il dataset dal file CSV
    dataset = pd.read_csv(nome_file)

    # Elenco delle variabili e i titoli dei grafici
    variabili = [
        ('school', 'Scuola di appartenenza'),
        ('age', 'Età degli studenti'),
        ('sex', 'Sesso'),
        ('address', 'Indirizzo - Urbano (U) o Rurale (R)'),
        ('famsize', 'Numero nucleo familiare: LE3 <= 3 | GT3 > 3'),
        ('Pstatus', 'Genitori vivono insieme?: Si (T) o No (A)'),
        ('Medu', 'Educazione madre'),
        ('Fedu', 'Educazione padre'),
        ('Mjob', 'Lavoro materno'),
        ('Fjob', 'Lavoro paterno'),
        ('reason', 'Motivo scelta scolastica'),
        ('guardian', 'Distribuzione tutore'),
        ('traveltime', 'Distanza dalla scuola (1: < 15 min, 2: 15-30 min...)'),
        ('studytime', 'Tempo studio weekend (1: < 2h, 2: 2-5h ...)'),
        ('failures', 'Numero classi non superate'),
        ('schoolsup', 'Supporto extra-scolastico'),
        ('famsup', 'Supporto familiare'),
        ('paid', 'Lezioni a pagamento'),
        ('activities', 'Attività extrascolastiche'),
        ('nursery', 'Asilo frequentato'),
        ('higher', 'Desiderio di istruzione superiore'),
        ('internet', 'Accesso a Internet'),
        ('romantic', 'Relazione romantica'),
        ('famrel', 'Qualità relazioni familiari'),
        ('freetime', 'Tempo libero dopo scuola'),
        ('goout', 'Uscite con amici'),
        ('Dalc', 'Consumo di alcol giorni feriali'),
        ('Walc', 'Consumo di alcol weekend'),
        ('health', 'Stato di salute'),
        ('absences', 'Assenze scolastiche raggruppate')
    ]

    # Definisci i bin per gli intervalli di assenze
    bins = [0, 5, 10, 15, 20, dataset['absences'].max() + 1]
    labels = ['0-5', '5-10', '10-15', '15-20', '20+']
    dataset['absences_grouped'] = pd.cut(dataset['absences'], bins=bins, labels=labels, right=False)

    # Organizza i grafici in 5 figure, ciascuna con 6 grafici
    num_figure = 5
    grafici_per_figure = 6

    for i in range(num_figure):
        # Crea una nuova figura per ogni gruppo di grafici
        fig, griglia = plt.subplots(2, 3, figsize=(18, 10))  # Layout 2x3
        griglia = griglia.flatten()  # Flatten per accesso semplificato agli assi

        # Itera sui grafici all'interno della figura corrente
        for j in range(grafici_per_figure):
            indice = i * grafici_per_figure + j
            if indice < len(variabili):
                colonna, titolo = variabili[indice]
                ax = griglia[j]

                # Plot in base al tipo di variabile
                if colonna == 'age':  # Grafico per distribuzione di 'age'
                    sns.histplot(dataset[colonna], bins=range(dataset[colonna].min(), dataset[colonna].max() + 1), kde=False, ax=ax)
                    ax.set_xticks(range(dataset[colonna].min(), dataset[colonna].max() + 1))
                elif colonna == 'absences':  # Grafico per assenze raggruppate
                    sns.countplot(x='absences_grouped', data=dataset, ax=ax)
                else:  # Grafico countplot per tutte le altre variabili
                    sns.countplot(x=colonna, data=dataset, ax=ax)

                ax.set_title(titolo)
                ax.set_xlabel(colonna.capitalize())
                ax.set_ylabel('Conteggio')

        # Aggiungi spazio tra i grafici
        plt.tight_layout(pad=3)

        # Salva la figura nella cartella specificata
        output_path = os.path.join(output_dir, f'figure_{i + 1}.png')
        plt.savefig(output_path)
        print(f'Figura salvata: {output_path}')
        plt.close(fig)  # Chiudi la figura per liberare memoria


