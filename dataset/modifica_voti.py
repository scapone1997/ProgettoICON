import pandas as pd

def modifica_voti(input_file, output_file):
    # Carica il dataset
    dataset = pd.read_csv(input_file)

    # Funzione per convertire i voti da una scala 0-20 a una scala 1-10
    def converti_voto(valore):
        nuovo_valore = valore / 2  # Conversione principale
        nuovo_valore = min(max(nuovo_valore, 1), 5)  # Assicura che il valore sia tra 1 e 10
        return round(nuovo_valore)  # Arrotonda al numero intero più vicino

    def converti_voto_categoriale(valore):
        """
        Converte un voto da 0-20 in una scala da 1 a 5:
        1: Non Sufficiente, 2: Sufficiente, 3: Buono, 4: Ottimo, 5: Eccellente
        """
        if 0 <= valore <= 9:
            return 1  # Non Sufficiente
        elif 10 <= valore <= 12:
            return 2  # Sufficiente
        elif 13 <= valore <= 15:
            return 3  # Buono
        elif 16 <= valore <= 17:
            return 4  # Ottimo
        elif 18 <= valore <= 20:
            return 5  # Eccellente
        else:
            return None  # Gestione di valori fuori dal range (se presente)

    # Applica la funzione a G1, G2 e G3
    dataset['G1'] = dataset['G1'].apply(converti_voto_categoriale)
    dataset['G2'] = dataset['G2'].apply(converti_voto_categoriale)
    dataset['G3'] = dataset['G3'].apply(converti_voto_categoriale)

    # Salva il dataset modificato in un nuovo file CSV
    dataset.to_csv(output_file, index=False)
    print(f"File salvato come {output_file}")



# Esegui la funzione con i file di input e output specificati
modifica_voti(
    "C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv",
    "C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat-scalaCat.csv"
)
