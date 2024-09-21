import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Definisci la funzione per convertire G3 in voti da 1 a 4
def convert_grade(g3):
    if g3 <= 12:
        return 1  # Fallimento
    else:
        return 0  # Successo

# Carica il dataset (assicurati che 'student-mat.csv' sia nella stessa cartella dello script)
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por.csv')

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

# Creazione e addestramento del modello di regressione lineare
model = LinearRegression()
model.fit(X_train, y_train)

# Predizione sul test set
y_pred = model.predict(X_test)

# Valutazione delle prestazioni del modello
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Errore quadratico medio (MSE):', mse)
print('Errore assoluto medio (MAE):', mae)
print('Coefficiente di determinazione (R^2):', r2)




