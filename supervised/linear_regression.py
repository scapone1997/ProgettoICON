from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from processing_dataset import prepara_dataset, separa_variabili

# Carica il dataset e trasforma le colonne categoriche
dataset_finale = prepara_dataset('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student_combined.csv')

# Seleziona solo le colonne di interesse per la regressione
colonne_interessate = ['Medu', 'Fedu'] + [col for col in dataset_finale.columns if col.startswith('famsize_') or col.startswith('address_') ] + ['G3']
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
print(f"(PRIMO) : Mean Squared Error (MSE): {mse}")
print(f"(PRIMO) : R² Score: {r2}")




