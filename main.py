import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# 1. Caricamento e preparazione dei dati
data = pd.read_csv('dataset.csv')

# Codifica delle variabili categoriche
label_encoders = {}
for column in ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Seleziona le variabili indipendenti e la variabile dipendente
X = data[['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'cogs', 'gross margin percentage']]
y = data['Total']

# Normalizzazione delle variabili indipendenti
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisione del dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Creazione e addestramento del modello di regressione lineare
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=X_train.shape[1], activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Addestramento del modello
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 3. Valutazione del modello
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')