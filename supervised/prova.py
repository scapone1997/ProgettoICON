import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Carica il dataset e trasforma le colonne categoriche
dataset = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat-scala10.csv')


# Separa le variabili indipendenti e la variabile target
X = dataset.drop(['G1', 'G2', 'G3'], axis=1)  # Escludi G1 e G2
Y = dataset['G3']

# Converti variabili categoriali in dummy variables
X = pd.get_dummies(X, drop_first=True)

# Divisione in train e test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scala i dati per uniformare le feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inizializza e addestra il modello
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, Y_train)

# Calcola l'importanza delle feature
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Stampa le feature più importanti
feature_names = X.columns
print("Feature importances:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]}")
