import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il tuo dataset (sostituisci con il percorso corretto del tuo file)
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por.csv')

# Stampa le colonne 'sex' e 'G3'
print(df[['sex', 'G3']])

# Plot della relazione tra 'sex' e 'G3' visualizzando tutti i valori di G3
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='G3', y='sex', palette=['lightcoral', 'royalblue'])

plt.title('Relazione Genere (sex) e Voto Finale (G3)')
plt.xlabel('Voto Finale (G3)')
plt.ylabel('Genere (sex)')
plt.grid(True)
plt.xticks(range(0, 21))
plt.show()