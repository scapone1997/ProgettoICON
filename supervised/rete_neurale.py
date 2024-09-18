import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. Carica il dataset
dataset = pd.read_csv("C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv")

# 2. Preprocessamento dei dati
# Converti tutte le colonne categoriche in dummy variables (one-hot encoding)
dataset = pd.get_dummies(dataset, drop_first=True)

# Separa le variabili indipendenti (X) da quella target (G3)
X = dataset.drop('G3', axis=1)  # Tutte le variabili eccetto G3
Y = dataset['G3']               # Variabile target G3

# 3. Divisione del dataset in training e test set (80% training, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Standardizzazione dei dati (normalizzazione delle feature)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Costruzione della rete neurale
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))  # Strato d'ingresso con 64 neuroni
model.add(Dense(32, activation='relu'))  # Strato nascosto con 32 neuroni
model.add(Dense(16, activation='relu'))  # Strato nascosto con 16 neuroni
model.add(Dense(1))  # Strato di output (predizione di G3, valore continuo)

# 6. Compilazione del modello
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Addestramento del modello
model.fit(X_train_scaled, Y_train, epochs=100, batch_size=16, verbose=1)

# 8. Usa SHAP per interpretare la rete neurale
# Crea un spiegatore SHAP
explainer = shap.KernelExplainer(model.predict, X_train_scaled)

# Calcola i valori SHAP per un sottoinsieme del test set (per questioni di tempo di calcolo)
shap_values = explainer.shap_values(X_test_scaled[:100])

# 9. Visualizza i risultati dell'importanza delle feature
shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X.columns)

# 10. Ordina e stampa le feature in base all'importanza media (mean absolute SHAP values)
shap_importance = np.mean(np.abs(shap_values), axis=0)
sorted_indices = np.argsort(shap_importance)[::-1]

print("Feature importance ranking (SHAP):")
for i in sorted_indices:
    print(f"{X.columns[i]}: {shap_importance[i]}")
