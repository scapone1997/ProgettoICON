import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Carica il tuo dataset (assicurati che il percorso sia corretto)
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv')

# Stampa le colonne 'health' e 'G3'
print(df[['health', 'G3']])

# Plot della correlazione tra 'health' e 'G3' con colori più contrastanti
plt.figure(figsize=(8, 6))

# Scatter plot con colori contrastanti
sns.scatterplot(data=df, x='health', y='G3', alpha=0.8, color='darkblue', edgecolor='black', s=100)

# Regressione con una linea di colore contrastante
sns.regplot(data=df, x='health', y='G3', scatter=False, color='darkorange', ci=None, x_jitter=0.1)

# Imposta i ticks sull'asse x per mostrare solo numeri interi
plt.xticks(range(df['health'].min(), df['health'].max() + 1))

plt.title('Relazione fra Salute e Voto Finale (G3)')
plt.xlabel('Salute (health)')
plt.ylabel('Voto Finale (G3)')
plt.grid(True)
plt.show()

