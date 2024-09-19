import pandas as pd


def modifica_assenze(input_file, output_file):
    # Carica il dataset dal file CSV
    dataset = pd.read_csv(input_file)

    # Funzione per convertire il numero di assenze secondo le regole specificate
    def converti_assenze(valore):
        if 0 <= valore <= 5:
            return 1
        elif 6 <= valore <= 10:
            return 2
        elif 11 <= valore <= 15:
            return 3
        elif 16 <= valore <= 20:
            return 4
        else:  # Valore > 20
            return 5

    # Applica la funzione alla colonna 'absences'
    dataset['absences'] = dataset['absences'].apply(converti_assenze)

    # Salva il dataset modificato in un nuovo file CSV
    dataset.to_csv(output_file, index=False)
    print(f"File salvato come {output_file}")


# Esempio di utilizzo della funzione
modifica_assenze(
    "C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat-scala10.csv",
    "C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat-scala10abs.csv"
)
