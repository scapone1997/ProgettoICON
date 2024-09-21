import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Caricamento dei dati
df_mat = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv')

# Creazione della variabile target (1 se G3 >= 17, altrimenti 0)
df_mat['high_grade'] = (df_mat['G3'] >= 15).astype(int)

# Trasforma la variabile categorica Pstatus in numerica binaria ('T' = 1, 'A' = 0)
df_mat['Pstatus'] = df_mat['Pstatus'].map({'T': 1, 'A': 0})
df_mat['address'] = df_mat['address'].map({'U': 1, 'R': 0})

# Seleziona le feature di interesse
features = df_mat[['famrel', 'Pstatus', 'address']]
target = df_mat['high_grade']

# Suddividi il dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Bilancia le classi nel training set usando SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Standardizza le feature numeriche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Calcola i pesi per le classi per dare più importanza alla classe minoritaria
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Modello Balanced Random Forest con parametri di base
model = BalancedRandomForestClassifier(random_state=42, class_weight=class_weights_dict)

# Imposta il range dei parametri per il tuning
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Randomized Search per l'ottimizzazione dei parametri
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=3, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train_balanced)

# Modello ottimizzato
best_model = random_search.best_estimator_

# Previsioni e valutazione
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

# Importanza delle feature
feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': best_model.feature_importances_
})
print(feature_importance.sort_values(by='Importance', ascending=False))
