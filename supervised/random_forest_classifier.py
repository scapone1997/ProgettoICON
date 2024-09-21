import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carica il dataset
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por-C.csv')

# Definisci la funzione per convertire G3 in voti da 1 a 3
def convert_grade(g3):
    if g3 <= 10:
        return 1
    else:
        return 2

# Applica la funzione alla colonna G3
df['G3_grade'] = df['G3'].apply(convert_grade)

# Seleziona le feature e la variabile target
features = ['address', 'famrel', 'Fedu']
X = df[features]
y = df['G3_grade']

# Codifica delle variabili categoriche
X = X.copy()
X['address'] = X['address'].map({'U': 0, 'R': 1})
# X['famsize'] = X['famsize'].map({'LE3': 0, 'GT3': 1})

# Divisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bilanciamento del dataset con SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Verifica la distribuzione delle classi dopo il bilanciamento
from collections import Counter
print('Distribuzione delle classi prima del bilanciamento:', Counter(y_train))
print('Distribuzione delle classi dopo il bilanciamento:', Counter(y_train_balanced))

# Modello di classificazione (ad esempio, Random Forest)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Predizione sul test set
y_pred = model.predict(X_test)

# Valutazione del modello
print('Accuratezza:', accuracy_score(y_test, y_pred))
print('\nReport di classificazione:\n', classification_report(y_test, y_pred))
print('Matrice di confusione:\n', confusion_matrix(y_test, y_pred))
