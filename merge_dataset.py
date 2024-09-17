import pandas as pd

# Leggi i due dataset (sostituisci con i percorsi corretti dei tuoi file CSV)
df_mat = pd.read_csv('student-mat.csv')
df_por = pd.read_csv('student-por.csv')

# Concatenare i due dataset (unione delle righe)
df_combined = pd.concat([df_mat, df_por], ignore_index=True)

# Salva il nuovo dataset in un file CSV
df_combined.to_csv('student_combined.csv', index=False)

# Stampa le prime righe del nuovo dataset per verifica
print(df_combined.head())