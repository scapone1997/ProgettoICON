import pandas as pd

# Carica il dataset (modifica il percorso del file se necessario)
dfMat = pd.read_csv('student-mat.csv')
dfCor = pd.read_csv('student-por.csv')

# Controlla la presenza di valori nulli per ogni colonna
null_values_mat = dfMat.isnull().sum()
null_values_cor = dfCor.isnull().sum()

# Mostra i risultati
print(null_values_cor)
print()
print(null_values_cor)