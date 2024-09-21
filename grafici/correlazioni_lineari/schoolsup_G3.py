import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il tuo dataset (sostituisci con il percorso corretto del tuo file)
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por.csv')

# Stampa le colonne 'schoolsup' e 'G3'
print(df[['schoolsup', 'G3']])

# Plot della relazione tra 'schoolsup' e 'G3' visualizzando tutti i valori di G3
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='G3', y='schoolsup', palette=['lightcoral', 'royalblue'])

plt.title('Relazione Supporto scolastico(schoolsup) e Voto Finale (G3)')
plt.xlabel('Voto Finale (G3)')
plt.ylabel('Supporto scolastico (schoolsup)')
plt.grid(True)
plt.xticks(range(0, 21))
plt.show()