import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Carica il tuo dataset (assicurati che il percorso sia corretto)
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por-C.csv')

# Stampa le colonne 'absences' e 'famrel'
print(df[['absences', 'famrel']])

# Plot della correlazione tra 'absences' e 'famrel' con colori più contrastanti
plt.figure(figsize=(8, 6))

# Scatter plot con colori contrastanti
sns.scatterplot(data=df, x='absences', y='famrel', alpha=0.8, color='darkblue', edgecolor='black', s=100)

# Regressione con una linea di colore contrastante
sns.regplot(data=df, x='absences', y='famrel', scatter=False, color='darkorange', ci=None, x_jitter=0.1)

# Imposta i ticks sull'asse x per mostrare solo numeri interi
plt.xticks(range(df['absences'].min(), df['absences'].max() + 1))

plt.title('Relazione fra Uscite e Rel. Familiari (famrel)')
plt.xlabel('Uscite (absences)')
plt.ylabel('Relazioni familiari (famrel)')
plt.grid(True)
plt.show()
