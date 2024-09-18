import pandas as pd


def modifica_voti(input_file, output_file):
    # Carica il dataset
    dataset = pd.read_csv(input_file)

    # Funzione per convertire i voti in base alle fasce di voto
    def converti_voto(valore):
        if valore >= 18:
            return 5
        elif valore >= 15:
            return 4
        elif valore >= 10:
            return 3
        else:
            return 2

    # Applica la funzione a G1, G2 e G3
    dataset['G1'] = dataset['G1'].apply(converti_voto)
    dataset['G2'] = dataset['G2'].apply(converti_voto)
    dataset['G3'] = dataset['G3'].apply(converti_voto)

    # Salva il dataset modificato in un nuovo file CSV
    dataset.to_csv(output_file, index=False)
    print(f"File salvato come {output_file}")


modifica_voti("C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat.csv", "C:/Users/simone.capone/PycharmProjects/ProgettoICON/dataset/student-mat-modificato.csv")
