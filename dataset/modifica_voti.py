import pandas as pd

def modifica_voti(input_file, output_file):
    # Carica il dataset
    dataset = pd.read_csv(input_file)

    # Funzione per convertire i voti da una scala 0-20 a una scala 1-10
    def converti_voto(valore):
        nuovo_valore = valore / 2  # Conversione principale
        nuovo_valore = min(max(nuovo_valore, 1), 10)  # Assicura che il valore sia tra 1 e 10
        return round(nuovo_valore)  # Arrotonda al numero intero più vicino

    # Applica la funzione a G1, G2 e G3
    dataset['G1'] = dataset['G1'].apply(converti_voto)
    dataset['G2'] = dataset['G2'].apply(converti_voto)
    dataset['G3'] = dataset['G3'].apply(converti_voto)

    # Salva il dataset modificato in un nuovo file CSV
    dataset.to_csv(output_file, index=False)
    print(f"File salvato come {output_file}")

# Esegui la funzione con i file di input e output specificati
modifica_voti(
    "C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv",
    "C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat-scala10.csv"
)
