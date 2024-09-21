import numpy as np
import pandas as pd

# Carica il dataset (assicurati di sostituire 'nome_del_tuo_file.csv' con il nome effettivo del tuo file)
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por.csv')

print(f"Numero righe dataset originale: {len(df)}")

# Trova le righe dove G2 == 0 e G3 == 0
righe_elim1 = df[((df['G2'] == 0) & (df['G3'] == 0))
                 | ((df['G1'] == 0) & (df['G3'] == 0))
                 | ((df['G1'] == 0) & (df['G3'] > 0))]

# Stampa il numero di righe che saranno eliminate
print(f"Numero di righe_elim1 da eliminare: {len(righe_elim1)}")
print("Righe elim1:")
print(righe_elim1)

df.drop(righe_elim1.index, inplace=True)
print(f"Numero righe dopo elim1: {len(df)}")


# Calcola la media di G1 e G2
media_G1_G2 = (df['G1'] + df['G2']) / 2

# Trova le righe da eliminare: (G1+G2)/2 > 0 e G3 == 0
condizione = (media_G1_G2 > 0) & (df['G3'] == 0)
righe_elim2 = df[condizione]

# Stampa il numero di righe che saranno eliminate
print(f"Numero di righe_elim2 da eliminare: {len(righe_elim2)}")
print("Righe elim2:")
print(righe_elim2)

df.drop(righe_elim2.index, inplace=True)
print(f"Numero righe dopo elim2: {len(df)}")

# media_G1_G2 = (df['G1'] + df['G2']) / 2
# differenza = abs(media_G1_G2 - df['G3'])
# print(differenza)
# differenza.to_csv('distanza.csv')

media_G1_G2 = (df['G1'] + df['G2']) / 2
differenza = abs(media_G1_G2 - df['G3'])
righe_distanza_grande = df[differenza > 2.5]
print(f"Numero di righe con differenza superiore a 3: {len(righe_distanza_grande)}")
print("Righe con differenza superiore a 3:")
print(righe_distanza_grande)

df.drop(righe_distanza_grande.index, inplace=True)
print(f"Numero righe dopo righe_distanza_grande: {len(df)}")

# Mostra il conteggio dei valori unici prima della sostituzione
print("Conteggio dei valori di 'traveltime' prima della sostituzione:")
print(df['traveltime'].value_counts())

# Sostituisci il valore 4 con 3 nella colonna 'traveltime'
df.loc[df['traveltime'] == 4, 'traveltime'] = 3

# Mostra il conteggio dei valori unici dopo la sostituzione
print("\nConteggio dei valori di 'traveltime' dopo la sostituzione:")
print(df['traveltime'].value_counts())

# Sostituisci il valore 4 con 3 nella colonna 'traveltime'
df.loc[df['famrel'] == 1, 'famrel'] = 2

righe_eta19 = df[df['age'] > 19]
print(f"Numero di righe con studenti con eta > 19: {len(righe_eta19)}")
print("Righe con eta > 19:")
print(righe_eta19)

df.drop(righe_eta19.index, inplace=True)
print(f"Numero righe dopo eta > 19: {len(df)}")

# righe_absencec = df[df['absences'] > 15]
# print(f"Numero righe assenze > 15: {len(righe_eta19)}")
# print("Righe righe assenze  > 15:")
# print(righe_absencec)
#
#
# df.drop(righe_absencec.index, inplace=True)
# print(f"Numero righe dopo assenze > 19: {len(df)}")


df.to_csv('student-por-C.csv', index=False)

