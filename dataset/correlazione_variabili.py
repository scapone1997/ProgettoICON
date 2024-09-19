import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il dataset (assicurati di avere il dataset corretto)
dataset = pd.read_csv("C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv")

# Crea una figura con 3 sottotrame in una griglia di 1 riga e 3 colonne
plt.figure(figsize=(18, 6))

# Prima trama: Relazione tra qualità delle relazioni familiari e consumo di alcol nei giorni feriali
plt.subplot(1, 3, 1)
sns.boxplot(x='famrel', y='Dalc', data=dataset)
plt.title('Famrel vs. Dalc')
plt.xlabel('Qualità Relazioni Familiari (1-5)')
plt.ylabel('Consumo di Alcol Giorni Feriali (1-5)')

# Seconda trama: Relazione tra qualità delle relazioni familiari e consumo di alcol nel weekend
plt.subplot(1, 3, 2)
sns.boxplot(x='famrel', y='Walc', data=dataset)
plt.title('Famrel vs. Walc')
plt.xlabel('Qualità Relazioni Familiari (1-5)')
plt.ylabel('Consumo di Alcol Weekend (1-5)')

# Terza trama: Relazione tra qualità delle relazioni familiari e voto finale
plt.subplot(1, 3, 3)
sns.scatterplot(x='famrel', y='G3', data=dataset)
plt.title('Famrel vs. G3')
plt.xlabel('Qualità Relazioni Familiari (1-5)')
plt.ylabel('Voto Finale (G3)')

# Mostra tutte le trame insieme in un'unica figura
plt.tight_layout()
plt.show()

