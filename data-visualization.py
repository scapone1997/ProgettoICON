import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leggi il dataset dal file CSV
dataset = pd.read_csv('student_combined.csv')

# Creare la prima figura con una griglia di 4 righe e 4 colonne (per i primi 16 grafici)
fig1, griglia_one = plt.subplots(4, 4, figsize=(20, 16))  # Layout 4x4

# PRIMA FIGURA (16 grafici)
# Grafico per la distribuzione di 'school'
sns.countplot(x='school', data=dataset, ax=griglia_one[0, 0])
griglia_one[0, 0].set_title('Distribuzione delle scuole')
griglia_one[0, 0].set_xlabel('Scuola')
griglia_one[0, 0].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'age'
sns.histplot(dataset['age'], bins=range(dataset['age'].min(), dataset['age'].max() + 1), kde=False, ax=griglia_one[0, 1])
griglia_one[0, 1].set_title('Distribuzione dell\'età degli studenti')
griglia_one[0, 1].set_xlabel('Età')
griglia_one[0, 1].set_ylabel('Conteggio')
griglia_one[0, 1].set_xticks(range(dataset['age'].min(), dataset['age'].max() + 1))

# Grafico per la distribuzione di 'sex'
sns.countplot(x='sex', data=dataset, ax=griglia_one[0, 2])
griglia_one[0, 2].set_title('Distribuzione per sesso')
griglia_one[0, 2].set_xlabel('Sesso')
griglia_one[0, 2].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'address'
sns.countplot(x='address', data=dataset, ax=griglia_one[0, 3])
griglia_one[0, 3].set_title('Distribuzione degli indirizzi')
griglia_one[0, 3].set_xlabel('Indirizzo')
griglia_one[0, 3].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'famsize'
sns.countplot(x='famsize', data=dataset, ax=griglia_one[1, 0])
griglia_one[1, 0].set_title('Distribuzione dimensioni familiari')
griglia_one[1, 0].set_xlabel('Dimensione familiare')
griglia_one[1, 0].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'Pstatus'
sns.countplot(x='Pstatus', data=dataset, ax=griglia_one[1, 1])
griglia_one[1, 1].set_title('Distribuzione dello stato abitativo')
griglia_one[1, 1].set_xlabel('Stato abitativo')
griglia_one[1, 1].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'Medu'
sns.countplot(x='Medu', data=dataset, ax=griglia_one[1, 2])
griglia_one[1, 2].set_title('Distribuzione educazione materna')
griglia_one[1, 2].set_xlabel('Educazione materna')
griglia_one[1, 2].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'Fedu'
sns.countplot(x='Fedu', data=dataset, ax=griglia_one[1, 3])
griglia_one[1, 3].set_title('Distribuzione educazione paterna')
griglia_one[1, 3].set_xlabel('Educazione paterna')
griglia_one[1, 3].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'Mjob'
sns.countplot(x='Mjob', data=dataset, ax=griglia_one[2, 0])
griglia_one[2, 0].set_title('Distribuzione del lavoro materno')
griglia_one[2, 0].set_xlabel('Lavoro materno')
griglia_one[2, 0].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'Fjob'
sns.countplot(x='Fjob', data=dataset, ax=griglia_one[2, 1])
griglia_one[2, 1].set_title('Distribuzione del lavoro paterno')
griglia_one[2, 1].set_xlabel('Lavoro paterno')
griglia_one[2, 1].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'reason'
sns.countplot(x='reason', data=dataset, ax=griglia_one[2, 2])
griglia_one[2, 2].set_title('Distribuzione del motivo per la scuola')
griglia_one[2, 2].set_xlabel('Motivo')
griglia_one[2, 2].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'guardian'
sns.countplot(x='guardian', data=dataset, ax=griglia_one[2, 3])
griglia_one[2, 3].set_title('Distribuzione del tutore')
griglia_one[2, 3].set_xlabel('Tutore')
griglia_one[2, 3].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'traveltime'
sns.countplot(x='traveltime', data=dataset, ax=griglia_one[3, 0])
griglia_one[3, 0].set_title('Distribuzione dei tempi di viaggio')
griglia_one[3, 0].set_xlabel('Tempo di viaggio')
griglia_one[3, 0].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'studytime'
sns.countplot(x='studytime', data=dataset, ax=griglia_one[3, 1])
griglia_one[3, 1].set_title('Distribuzione del tempo di studio')
griglia_one[3, 1].set_xlabel('Tempo di studio')
griglia_one[3, 1].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'failures'
sns.countplot(x='failures', data=dataset, ax=griglia_one[3, 2])
griglia_one[3, 2].set_title('Distribuzione degli insuccessi scolastici')
griglia_one[3, 2].set_xlabel('Insuccessi')
griglia_one[3, 2].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'schoolsup'
sns.countplot(x='schoolsup', data=dataset, ax=griglia_one[3, 3])
griglia_one[3, 3].set_title('Supporto scolastico')
griglia_one[3, 3].set_xlabel('Supporto scolastico')
griglia_one[3, 3].set_ylabel('Conteggio')

# Aggiungere un po' di spazio tra i grafici per la prima figura
plt.tight_layout(pad = 3)

# Creare la seconda figura con una griglia di 2 righe e 5 colonne (per i restanti 9 grafici)
fig2, griglia_two = plt.subplots(3, 5, figsize=(25, 12))  # Layout 2x5

# SECONDA FIGURA (9 grafici)
# Grafico per la distribuzione di 'famsup'
sns.countplot(x='famsup', data=dataset, ax=griglia_two[0,0])
griglia_two[0, 0].set_title('Supporto familiare')
griglia_two[0, 0].set_xlabel('Supporto familiare')
griglia_two[0, 0].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'paid'
sns.countplot(x='paid', data=dataset, ax=griglia_two[0, 1])
griglia_two[0, 1].set_title('Lezioni a pagamento')
griglia_two[0, 1].set_xlabel('Lezioni a pagamento')
griglia_two[0, 1].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'activities'
sns.countplot(x='activities', data=dataset, ax=griglia_two[0, 2])
griglia_two[0, 2].set_title('Attività extrascolastiche')
griglia_two[0, 2].set_xlabel('Attività')
griglia_two[0, 2].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'nursery'
sns.countplot(x='nursery', data=dataset, ax=griglia_two[0, 3])
griglia_two[0, 3].set_title('Asilo frequentato')
griglia_two[0, 3].set_xlabel('Asilo')
griglia_two[0, 3].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'higher'
sns.countplot(x='higher', data=dataset, ax=griglia_two[0, 4])
griglia_two[0, 4].set_title('Desiderio di istruzione superiore')
griglia_two[0, 4].set_xlabel('Istruzione superiore')
griglia_two[0, 4].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'internet'
sns.countplot(x='internet', data=dataset, ax=griglia_two[1, 0])
griglia_two[1, 0].set_title('Accesso a Internet')
griglia_two[1, 0].set_xlabel('Internet')
griglia_two[1, 0].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'romantic'
sns.countplot(x='romantic', data=dataset, ax=griglia_two[1, 1])
griglia_two[1, 1].set_title('Relazione romantica')
griglia_two[1, 1].set_xlabel('Romantico')
griglia_two[1, 1].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'famrel'
sns.countplot(x='famrel', data=dataset, ax=griglia_two[1,2])
griglia_two[1, 2].set_title('Relazioni familiari')
griglia_two[1, 2].set_xlabel('Relazione familiare')
griglia_two[1, 2].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'freetime'
sns.countplot(x='freetime', data=dataset, ax=griglia_two[1, 3])
griglia_two[1, 3].set_title('Tempo libero')
griglia_two[1, 3].set_xlabel('Tempo libero')
griglia_two[1, 3].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'goout'
sns.countplot(x='goout', data=dataset, ax=griglia_two[1,4])
griglia_two[1, 4].set_title('Uscite con amici')
griglia_two[1, 4].set_xlabel('Uscire con amici')
griglia_two[1, 4].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'Dalc' (alcol nei giorni feriali)
sns.countplot(x='Dalc', data=dataset, ax=griglia_two[2,0])
griglia_two[2, 0].set_title('Consumo di alcol nei giorni feriali')
griglia_two[2, 0].set_xlabel('Consumo di alcol (giorni feriali)')
griglia_two[2, 0].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'Walc' (alcol nel weekend)
sns.countplot(x='Walc', data=dataset, ax=griglia_two[2, 1])
griglia_two[2, 1].set_title('Consumo di alcol nel weekend')
griglia_two[2, 1].set_xlabel('Consumo di alcol (weekend)')
griglia_two[2, 1].set_ylabel('Conteggio')

# Grafico per la distribuzione di 'health'
sns.countplot(x='health', data=dataset, ax=griglia_two[2,2])
griglia_two[2, 2].set_title('Stato di salute')
griglia_two[2, 2].set_xlabel('Salute attuale')
griglia_two[2, 2].set_ylabel('Conteggio')

plt.tight_layout(pad=3)

# TERZA FIGURA (1 grafico)
fig3, griglia_three = plt.subplots(figsize=(10, 6))

# Grafico per la distribuzione di 'absences'
sns.histplot(dataset['absences'], bins=range(0, dataset['absences'].max() + 1), kde=False, ax=griglia_three)
griglia_three.set_title('Assenze scolastiche')
griglia_three.set_xlabel('Assenze')
griglia_three.set_ylabel('Conteggio')
griglia_three.set_xticks(range(0, dataset['absences'].max() + 1, 5))

# Aggiungere un po' di spazio tra i grafici per la seconda figura


# Mostrare i grafici
plt.show()
