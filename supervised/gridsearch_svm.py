import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Carica il dataset
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por-C.csv')

# Definisci la funzione per convertire G3 in voti da 0 (successo) a 1 (fallimento)
def convert_grade(g3):
    if g3 <= 14:
        return 1  # Fallimento
    else:
        return 0  # Successo

# Applica la funzione alla colonna G3
df['G3_grade'] = df['G3'].apply(convert_grade)

# Seleziona le feature e la variabile target
features = ['address', 'famrel', 'Fedu']
X = df[features]
y = df['G3_grade']

# Codifica delle variabili categoriali
X['address'] = X['address'].map({'U': 0, 'R': 1})

# Standardizzazione delle feature
scaler = StandardScaler()
X[['famrel', 'Fedu']] = scaler.fit_transform(X[['famrel', 'Fedu']])

# Divisione del dataset in training set e test set con stratificazione
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Bilanciamento del dataset con SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Definizione della griglia dei parametri per la SVC
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Ricerca dei migliori parametri con GridSearchCV
grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=2, n_jobs=-1, cv=3)
grid.fit(X_train_balanced, y_train_balanced)

# Migliori parametri trovati
print("Migliori Parametri:", grid.best_params_)

# Usa il miglior modello trovato
best_model = grid.best_estimator_

# Configura Stratified K-Fold per la cross-validation con 3 fold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Esegui la cross-validation per calcolare l'accuratezza media e deviazione standard
scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=skf, scoring='accuracy')
print(f"Accuratezza media con cross-validation: {np.mean(scores):.2f}, Deviazione standard: {np.std(scores):.2f}")

# Predizione sul test set
y_pred = best_model.predict(X_test)

# Valutazione del modello sul test set
accuracy = accuracy_score(y_test, y_pred)
accuracy_rounded = math.ceil(accuracy * 100) / 100
print('Accuratezza sul test set:', accuracy_rounded)
print('Report di classificazione:\n', classification_report(y_test, y_pred))
print('Matrice di confusione:\n', confusion_matrix(y_test, y_pred))
