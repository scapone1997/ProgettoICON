import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il dataset dal file CSV
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student_combined.csv')

# Mappare le variabili categoriche binarie su valori numerici
binary_mappings = {
    'school': {'GP': 0, 'MS': 1},
    'sex': {'F': 0, 'M': 1},
    'address': {'R': 0, 'U': 1},
    'famsize': {'GT3': 0, 'LE3': 1},
    'Pstatus': {'T': 0, 'A': 1},
    'schoolsup': {'yes': 1, 'no': 0},
    'famsup': {'yes': 1, 'no': 0},
    'paid': {'yes': 1, 'no': 0},
    'activities': {'yes': 1, 'no': 0},
    'nursery': {'yes': 1, 'no': 0},
    'higher': {'yes': 1, 'no': 0},
    'internet': {'yes': 1, 'no': 0},
    'romantic': {'yes': 1, 'no': 0}
}

# Applicare le mappature binarie
df.replace(binary_mappings, inplace=True)

# Convertire le variabili categoriche con più classi usando One-Hot Encoding
df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)

# Calcolare la matrice di correlazione
correlation_matrix = df.corr()

# Visualizzare la matrice di correlazione con una heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice di Correlazione')
plt.show()
