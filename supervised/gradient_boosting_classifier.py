import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carica il dataset
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por-C.csv')

# Definisci la funzione per convertire G3 in voti da 1 a 0
def convert_grade(g3):
    if g3 <= 10:
        return 1  # Fallimento
    else:
        return 0  # Successo

# Applica la funzione alla colonna G3
df['G3_grade'] = df['G3'].apply(convert_grade)

# Seleziona le feature e la variabile target
features = ['address', 'famrel', 'Fedu', 'Medu']
X = df[features]
y = df['G3_grade']

# Codifica delle variabili categoriali
X = X.copy()
X['address'] = X['address'].map({'U': 0, 'R': 1})

# Divisione del dataset in training set e test set con stratificazione
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Bilanciamento del dataset con SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Modello di classificazione - Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Predizione sul test set
y_pred = model.predict(X_test)

# Valutazione del modello
accuracy = accuracy_score(y_test, y_pred)
accuracy_rounded = math.ceil(accuracy * 100) / 100
print('Accuratezza:', accuracy_rounded)
print('Report di classificazione:\n', classification_report(y_test, y_pred))
print('Matrice di confusione:\n', confusion_matrix(y_test, y_pred))
