import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Carica il tuo dataset (assicurati che il percorso sia corretto)
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por.csv')

# Stampa le colonne 'Dalc' e 'G3'
print(df[['Dalc', 'G3']])

# Plot della correlazione tra 'Dalc' e 'G3' con colori più contrastanti
plt.figure(figsize=(8, 6))

# Scatter plot con colori contrastanti
sns.scatterplot(data=df, x='Dalc', y='G3', alpha=0.8, color='darkblue', edgecolor='black', s=100)

# Regressione con una linea di colore contrastante
sns.regplot(data=df, x='Dalc', y='G3', scatter=False, color='darkorange', ci=None, x_jitter=0.1)

# Imposta i ticks sull'asse x per mostrare solo numeri interi
plt.xticks(range(df['Dalc'].min(), df['Dalc'].max() + 1))

plt.title('Relazione fra Alcol (G. Feriali) e Voto Finale (G3)')
plt.xlabel('Consumo di alcol nei giorno lavorativi.')
plt.ylabel('Voto Finale (G3)')
plt.grid(True)
plt.show()

