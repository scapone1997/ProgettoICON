import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Carica il dataset
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por-C.csv')

# Definisci la funzione per convertire G3 in voti da 1 a 2
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

# Codifica delle variabili categoriche
X['address'] = X['address'].map({'U': 0, 'R': 1})

# Divisione del dataset in training set e test set con stratificazione
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Bilanciamento del dataset con SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Modello di classificazione - Regressione Logistica
model = LogisticRegression(max_iter=1000, random_state=42)

# Configura Stratified K-Fold per la cross-validation con 3 fold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Esegui la cross-validation per calcolare l'accuratezza
scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=skf, scoring='accuracy')

# Stampa i risultati della cross-validation
print(f"Accuratezza media con cross-validation: {np.mean(scores):.2f}, Deviazione standard: {np.std(scores):.2f}")

# Addestra il modello con tutti i dati di addestramento bilanciati
model.fit(X_train_balanced, y_train_balanced)

# Predizione sul test set
y_pred = model.predict(X_test)

# Valutazione del modello sul test set
accuracy = accuracy_score(y_test, y_pred)
accuracy_rounded = math.ceil(accuracy * 100) / 100
print('Accuratezza sul test set:', accuracy_rounded)
print('Report di classificazione:\n', classification_report(y_test, y_pred))
print('Matrice di confusione:\n', confusion_matrix(y_test, y_pred))
