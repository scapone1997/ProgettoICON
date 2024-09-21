import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Caricare il Dataset
df = pd.read_csv('C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-por-C.csv')

# 2a. Separare le feature dalla variabile target
X = df.drop('G3', axis=1)  # Tutte le colonne eccetto 'G3'
y = df['G3']               # Variabile target

# 2b. Codificare le variabili categoriali
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 2c. Gestire i valori mancanti (se presenti)
X = X.fillna(X.median())

# 3. Dividere il dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Addestrare il modello Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Ottenere e ordinare le importanze delle feature
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# 6. Visualizzare le importanze delle feature
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
plt.title('Importanza delle Feature nella Predizione di G3')
plt.xlabel('Importanza')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Stampa delle feature ordinate per importanza
print(feature_importances)
