import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leggi il dataset dal file CSV (sostituisci 'student_combined.csv' con il percorso corretto del file)
df = pd.read_csv('student_combined.csv')

# Creare una figura con una griglia di 2 righe e 2 colonne per i 4 grafici
fig, axes = plt.subplots(2, 4, figsize=(12, 10))  # Layout 2x2, dimensione aumentata per chiarezza

# Grafico per la distribuzione di 'school' (grafico a barre)
sns.countplot(x='school', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Distribuzione delle scuole')
axes[0, 0].set_xlabel('Scuola')
axes[0, 0].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'age' (istogramma)
sns.histplot(df['age'], bins=range(df['age'].min(), df['age'].max() + 1), kde=False, ax=axes[0, 1])
axes[0, 1].set_title('Distribuzione dell\'età degli studenti')
axes[0, 1].set_xlabel('Età')
axes[0, 1].set_ylabel('Conteggio')
axes[0, 1].set_xticks(range(df['age'].min(), df['age'].max() + 1))  # Mostra solo valori interi sull'asse X

# Grafico per la distribuzione di 'sex' (grafico a barre)
sns.countplot(x='sex', data=df, ax=axes[1, 0])
axes[0, 2].set_title('Distribuzione per sesso')
axes[0, 2].set_xlabel('Sesso')
axes[0, 2].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'address' (grafico a barre)
sns.countplot(x='address', data=df, ax=axes[1, 1])
axes[0, 3].set_title('Distribuzione degli indirizzi')
axes[0, 3].set_xlabel('Indirizzo')
axes[0, 3].set_ylabel('Conteggio')

# Aggiungere un po' di spazio tra i grafici
plt.tight_layout()

# Mostrare i grafici
plt.show()