import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_jobs_vs_G3(df):
    """
    Funzione che calcola la correlazione tra il lavoro dei genitori (Mjob e Fjob) e il voto finale G3.
    Mostra due grafici affiancati che evidenziano l'impatto delle professioni dei genitori sui risultati scolastici.
    I grafici sono ordinati con le stesse categorie su entrambi gli assi x.
    """
    # Ordine delle categorie da mantenere uguale per entrambi i grafici
    job_order = ['teacher', 'health', 'services', 'at_home', 'other']

    plt.figure(figsize=(14, 6))

    # Grafico 1: Relazione tra Mjob e G3
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='Mjob', y='G3', order=job_order, palette='viridis')
    plt.title('Relazione tra Lavoro della Madre (Mjob) e Voto Finale (G3)')
    plt.xlabel('Lavoro della Madre (Mjob)')
    plt.ylabel('Voto Finale (G3)')
    plt.xticks(rotation=45)

    # Grafico 2: Relazione tra Fjob e G3
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='Fjob', y='G3', order=job_order, palette='viridis')
    plt.title('Relazione tra Lavoro del Padre (Fjob) e Voto Finale (G3)')
    plt.xlabel('Lavoro del Padre (Fjob)')
    plt.ylabel('Voto Finale (G3)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# Carica il dataset e utilizza la funzione
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv')
plot_correlation_jobs_vs_G3(df)