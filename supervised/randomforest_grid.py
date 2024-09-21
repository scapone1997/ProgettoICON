import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Carica il dataset
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por-C.csv')

# Definisci la funzione per convertire G3 in voti da 1 a 2
def convert_grade(g3):
    if g3 <= 12:
        return 1  # Fallimento
    else:
        return 0  # Successo

# Applica la funzione alla colonna G3
df['G3_grade'] = df['G3'].apply(convert_grade)

# Seleziona le feature e la variabile target
features = ['famrel', 'address', 'Fedu']
X = df[features]
y = df['G3_grade']

# Codifica delle variabili categoriche
X['address'] = X['address'].map({'U': 0, 'R': 1})

# Divisione del dataset in training set e test set con stratificazione
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Bilanciamento del dataset con SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Creazione del pipeline con scaling e regressione logistica
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaler per le feature numeriche
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))  # Regressione logistica
])

# Definizione dei parametri per Grid Search
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],  # Regolarizzazione
    'logreg__penalty': ['l1', 'l2'],       # Tipo di regolarizzazione
    'logreg__solver': ['liblinear', 'saga'] # Solvers adatti per 'l1' e 'l2'
}

# Ricerca dei migliori parametri con GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Migliori parametri trovati
print("Migliori Parametri:", grid_search.best_params_)

# Uso del modello ottimizzato per fare previsioni
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Valutazione del modello ottimizzato
print('Accuratezza:', accuracy_score(y_test, y_pred))
print('Report di classificazione:\n', classification_report(y_test, y_pred))
print('Matrice di confusione:\n', confusion_matrix(y_test, y_pred))
