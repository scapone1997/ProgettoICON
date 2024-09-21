import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il tuo dataset (sostituisci con il percorso corretto del tuo file)
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por.csv')

# Stampa le colonne 'Pstatus' e 'G3'
print(df[['Pstatus', 'G3']])

# Plot della relazione tra 'Pstatus' e 'G3' visualizzando tutti i valori di G3
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='G3', y='Pstatus', palette=['lightcoral', 'royalblue'])

plt.title('Relazione Stato abitativo dei genitori (Pstatus) e Voto Finale (G3)')
plt.xlabel('Voto Finale (G3)')
plt.ylabel('Genitori vivono insieme? (Pstatus)')
plt.grid(True)
plt.xticks(range(0, 21))
plt.show()

# Gli studenti urbani tendono ad avere voti medi più alti (concentrati tra 10 e 14),
# mentre quelli rurali hanno voti più variabili e mediamente più bassi (tra 8 e 11).
# Il grafico evidenzia anche alcuni outlier, indicando
# casi di voti estremamente bassi o alti in entrambe le categorie.