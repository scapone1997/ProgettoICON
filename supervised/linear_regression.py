import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from processing_dataset import prepara_dataset, separa_variabili

# Carica il dataset e trasforma le colonne categoriche
dataset_finale = prepara_dataset('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv')

# Seleziona solo le colonne di interesse per la regressione
colonne_interessate = ['studytime', 'absences'] + ['G1'] + ['G3']
dataset_regressione = dataset_finale[colonne_interessate]

# Verifica dei valori mancanti
print(dataset_regressione)

# Statistiche descrittive del dataset
print(dataset_regressione.describe())

print("\n")

# Separo le variabili X dalla variabile target G3 .
X, Y = separa_variabili(dataset_regressione)

# 1. Dividi il dataset in training e test set (80% training, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 2. Standardizza le variabili indipendenti
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Crea e addestra il modello di regressione lineare
modello = LinearRegression()
modello.fit(X_train_scaled, Y_train)

# 4. Fai previsioni sui dati di test
Y_pred = modello.predict(X_test_scaled)

# 5. Valuta il modello
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Stampa i risultati
print(f"(LINEAR REGRESSION) : Mean Squared Error (MSE): {mse}")
print(f"(LINEAR REGRESSION) : R² Score: {r2}")

modello = DecisionTreeRegressor(random_state=42)
modello.fit(X_train_scaled, Y_train)

Y_pred = modello.predict(X_test_scaled)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Stampa i risultati
print(f"(DECISION TREE REGRESSION) : Mean Squared Error (MSE): {mse}")
print(f"(DECISION TREE REGRESSION) : R² Score: {r2}")

# 3. Crea e addestra il modello Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, Y_train)

# 4. Fai previsioni sui dati di test
Y_pred_gb = gb_model.predict(X_test_scaled)

# 5. Valuta il modello
mse = mean_squared_error(Y_test, Y_pred_gb)
r2 = r2_score(Y_test, Y_pred_gb)

# Stampa i risultati
print(f"(GRADIENT BOOSTING) : Mean Squared Error (MSE): {mse}")
print(f"(GRADIENT BOOSTING) : R² Score: {r2}")


# # Definisci il modello Random Forest
# rf_model = RandomForestRegressor(random_state=42)
#
# # Definisci la griglia di parametri da testare
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None]
# }
#
# # Configura GridSearchCV
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#
# # Esegui GridSearchCV
# grid_search.fit(X_train_scaled, Y_train)
#
# # Ottieni il miglior modello
# best_rf_model = grid_search.best_estimator_
#
# # Fai previsioni
# Y_pred_rf = best_rf_model.predict(X_test_scaled)
#
# # Valuta il miglior modello
# mse_rf = mean_squared_error(Y_test, Y_pred_rf)
# r2_rf = r2_score(Y_test, Y_pred_rf)
#
# print(f"Random Forest Mean Squared Error (MSE): {mse_rf}")
# print(f"Random Forest R² Score: {r2_rf}")





