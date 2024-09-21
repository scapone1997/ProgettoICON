import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Caricamento dei dati
df_mat = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por.csv')

# Creazione della variabile target (1 se G3 >= 17, altrimenti 0)
df_mat['high_grade'] = (df_mat['G3'] >= 17).astype(int)

# Trasforma le variabili categoriali in numeriche binarie
df_mat['address'] = df_mat['address'].map({'U': 1, 'R': 0})
df_mat['famsize'] = df_mat['famsize'].map({'LE3': 1, 'GT3': 0})
df_mat['internet'] = df_mat['internet'].map({'yes': 1, 'no': 0})

# Seleziona le feature di interesse
features = df_mat[['address', 'famsize', 'famrel', 'internet']]
target = df_mat['high_grade']

# Suddividi il dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Bilancia le classi nel training set usando SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_train_balanced, y_train_balanced = smoteenn.fit_resample(X_train, y_train)

# Standardizza le feature numeriche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Inizializza il modello Random Forest con pesi bilanciati
model = RandomForestClassifier(class_weight='balanced', random_state=42)#Do alla classe minoritaria un peso maggiore

# Definisci i parametri per la ricerca RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],          # Numero di alberi nella foresta; più alberi possono migliorare la stabilità e l'accuratezza.
    'max_depth': [None, 10, 20, 30, 40],          # Massima profondità degli alberi; None significa nessun limite, alberi più profondi possono catturare più complessità.
    'min_samples_split': [2, 5, 10, 15],          # Numero minimo di campioni richiesti per dividere un nodo; valori più alti possono ridurre l'overfitting.
    'min_samples_leaf': [1, 2, 4, 6],             # Numero minimo di campioni richiesti per essere un nodo foglia; valori più alti possono portare a modelli più semplici.
    'max_features': ['sqrt', 'log2'],             # Numero massimo di feature da considerare per la divisione; 'sqrt' e 'log2' bilanciano complessità e prestazioni.
    'bootstrap': [True, False]                    # Se campionare i dati con sostituzione; True può ridurre la varianza del modello.
}

# Ottimizzazione dei parametri usando RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train_balanced)

# Modello ottimizzato
best_model = random_search.best_estimator_

# Valutazione del modello con Cross-Validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train_balanced, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores)}")

# Previsioni e valutazione
y_pred = best_model.predict(X_test_scaled)
print(f"Best Parameters: {random_search.best_params_}")
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

# Analisi delle feature importanti
importances = best_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(feature_importance)

#Classe 0 studenti con voto finale minore di 17.
#Classe 1 studenti con voto finale uguale o superiore a 17.
