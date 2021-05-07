# Implemented by customization of code available at: https://github.com/tlawrence3/amPEPpy

import argparse
import numpy as np
import pandas as pd
import sklearn.utils
from sklearn.ensemble import RandomForestClassifier
from Bio import SeqIO
import re


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", "-t", help="path to training set")
    parser.add_argument("--test", "-e", help="path to test set")
    parser.add_argument("--output", "-o", help="path to save results")

    args = parser.parse_args()

    return args


def score(seqs):
    CTD = {'hydrophobicity': {1: ['R', 'K', 'E', 'D', 'Q', 'N'], 2: ['G', 'A', 'S', 'T', 'P', 'H', 'Y'],
                              3: ['C', 'L', 'V', 'I', 'M', 'F', 'W']},
           'normalized.van.der.waals': {1: ['G', 'A', 'S', 'T', 'P', 'D', 'C'], 2: ['N', 'V', 'E', 'Q', 'I', 'L'],
                                        3: ['M', 'H', 'K', 'F', 'R', 'Y', 'W']},
           'polarity': {1: ['L', 'I', 'F', 'W', 'C', 'M', 'V', 'Y'], 2: ['P', 'A', 'T', 'G', 'S'],
                        3: ['H', 'Q', 'R', 'K', 'N', 'E', 'D']},
           'polarizability': {1: ['G', 'A', 'S', 'D', 'T'], 2: ['C', 'P', 'N', 'V', 'E', 'Q', 'I', 'L'],
                              3: ['K', 'M', 'H', 'F', 'R', 'Y', 'W']},
           'charge': {1: ['K', 'R'],
                      2: ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
                      3: ['D', 'E']},
           'secondary': {1: ['E', 'A', 'L', 'M', 'Q', 'K', 'R', 'H'], 2: ['V', 'I', 'Y', 'C', 'W', 'F', 'T'],
                         3: ['G', 'N', 'P', 'S', 'D']},
           'solvent': {1: ['A', 'L', 'F', 'C', 'G', 'I', 'V', 'W'], 2: ['R', 'K', 'Q', 'E', 'N', 'D'],
                       3: ['M', 'S', 'P', 'T', 'H', 'Y']}}
    header = []
    groups = [1, 2, 3]
    values = [0, 25, 50, 75, 100]
    for AAproperty in CTD:
        for types in groups:
            for numbers in values:
                label = ""
                label = label.join("{}.{}.{}".format(AAproperty, types, numbers))
                header.append(label)
    All_groups = []
    Sequence_names = []
    for sequences in seqs:
        sequence_name = sequences.id
        Sequence_names.append(sequence_name)
        sequence = str(sequences.seq)
        sequence = sequence.replace("*", "")
        sequencelength = len(sequence)
        Sequence_group = []
        for AAproperty in CTD:
            propvalues = ""
            for letter in sequence:
                if letter in CTD[AAproperty][1]:
                    propvalues += "1"
                elif letter in CTD[AAproperty][2]:
                    propvalues += "2"
                elif letter in CTD[AAproperty][3]:
                    propvalues += "3"
            abpos_1 = [i for i in range(len(propvalues)) if propvalues.startswith("1", i)]
            abpos_1 = [x + 1 for x in abpos_1]
            abpos_1.insert(0, "-")
            abpos_2 = [i for i in range(len(propvalues)) if propvalues.startswith("2", i)]
            abpos_2 = [x + 1 for x in abpos_2]
            abpos_2.insert(0, "-")
            abpos_3 = [i for i in range(len(propvalues)) if propvalues.startswith("3", i)]
            abpos_3 = [x + 1 for x in abpos_3]
            abpos_3.insert(0, "-")
            property_group1_length = propvalues.count("1")
            if property_group1_length == 0:
                Sequence_group.extend([0, 0, 0, 0, 0])
            elif property_group1_length == 1:
                Sequence_group.append((abpos_1[1] / sequencelength) * 100)
                Sequence_group.append((abpos_1[1] / sequencelength) * 100)
                Sequence_group.append((abpos_1[1] / sequencelength) * 100)
                Sequence_group.append((abpos_1[1] / sequencelength) * 100)
                Sequence_group.append((abpos_1[1] / sequencelength) * 100)
            elif property_group1_length == 2:
                Sequence_group.append((abpos_1[1] / sequencelength) * 100)
                Sequence_group.append((abpos_1[1] / sequencelength) * 100)
                Sequence_group.append((abpos_1[round((0.5 * property_group1_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_1[round((0.75 * property_group1_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_1[property_group1_length] / sequencelength) * 100)
            else:
                Sequence_group.append((abpos_1[1] / sequencelength) * 100)
                Sequence_group.append((abpos_1[round((0.25 * property_group1_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_1[round((0.5 * property_group1_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_1[round((0.75 * property_group1_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_1[property_group1_length] / sequencelength) * 100)

            property_group2_length = propvalues.count("2")
            if property_group2_length == 0:
                Sequence_group.extend([0, 0, 0, 0, 0])
            elif property_group2_length == 1:
                Sequence_group.append((abpos_2[1] / sequencelength) * 100)
                Sequence_group.append((abpos_2[1] / sequencelength) * 100)
                Sequence_group.append((abpos_2[1] / sequencelength) * 100)
                Sequence_group.append((abpos_2[1] / sequencelength) * 100)
                Sequence_group.append((abpos_2[1] / sequencelength) * 100)
            elif property_group2_length == 2:
                Sequence_group.append((abpos_2[1] / sequencelength) * 100)
                Sequence_group.append((abpos_2[1] / sequencelength) * 100)
                Sequence_group.append((abpos_2[round((0.5 * property_group2_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_2[round((0.75 * property_group2_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_2[property_group2_length] / sequencelength) * 100)
            else:
                Sequence_group.append((abpos_2[1] / sequencelength) * 100)
                Sequence_group.append((abpos_2[round((0.25 * property_group2_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_2[round((0.5 * property_group2_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_2[round((0.75 * property_group2_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_2[property_group2_length] / sequencelength) * 100)

            property_group3_length = propvalues.count("3")
            if property_group3_length == 0:
                Sequence_group.extend([0, 0, 0, 0, 0])
            elif property_group3_length == 1:
                Sequence_group.append((abpos_3[1] / sequencelength) * 100)
                Sequence_group.append((abpos_3[1] / sequencelength) * 100)
                Sequence_group.append((abpos_3[1] / sequencelength) * 100)
                Sequence_group.append((abpos_3[1] / sequencelength) * 100)
                Sequence_group.append((abpos_3[1] / sequencelength) * 100)
            elif property_group3_length == 2:
                Sequence_group.append((abpos_3[1] / sequencelength) * 100)
                Sequence_group.append((abpos_3[1] / sequencelength) * 100)
                Sequence_group.append((abpos_3[round((0.5 * property_group3_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_3[round((0.75 * property_group3_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_3[property_group3_length] / sequencelength) * 100)
            else:
                Sequence_group.append((abpos_3[1] / sequencelength) * 100)
                Sequence_group.append((abpos_3[round((0.25 * property_group3_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_3[round((0.5 * property_group3_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_3[round((0.75 * property_group3_length) - 0.1)] / sequencelength) * 100)
                Sequence_group.append((abpos_3[property_group3_length] / sequencelength) * 100)
        All_groups.append(Sequence_group)

    Property_dataframe = pd.DataFrame.from_dict(All_groups)
    Property_dataframe.columns = header
    Property_dataframe.index = Sequence_names
    # Property_dataframe.to_csv(f'{date.today().strftime("%Y%m%d")}_scored_features.csv', index=True, index_label="sequence_name")
    return Property_dataframe


def main():
    args = parse_arguments()

    seqs_pos = (s for s in SeqIO.parse(args.train, "fasta") if re.search("AMP=1", s.id))
    seqs_neg = (s for s in SeqIO.parse(args.train, "fasta") if re.search("AMP=0", s.id))

    positive_df = score(seqs_pos)
    positive_df['classi'] = 1

    negative_df = score(seqs_neg)
    negative_df['classi'] = 0

    feature_drop_list = []

    feature_drop_list.append("classi")
    training_df = pd.concat([positive_df, negative_df])
    training_df = sklearn.utils.shuffle(training_df, random_state=2012) # random seed was set to 2012 for all of the analyses for the manuscript
    X = training_df.drop(columns=feature_drop_list)
    y = training_df.classi
    clf = RandomForestClassifier(n_estimators=160, # Default is 160 which was shown to produce the lowest out of bag error on training data
                                 oob_score=False,
                                 random_state=2012)
    clf.fit(X, y)


    classify_df = score(SeqIO.parse(args.test, "fasta"))
    id_info = classify_df.index.tolist()
    targets = []
    for id in id_info:
        if re.search("AMP=1", id):
            targets.append(1)
        elif re.search("AMP=0", id):
            targets.append(0)

    preds = clf.predict_proba(classify_df)

    results = pd.DataFrame({
        'ID': id_info,
        'target': targets,
        'prediction': np.rint(preds[:,1]).astype(np.int32).flatten().tolist(),
        'probability': preds[:,1].tolist()},
    ).reset_index(drop=True)

    results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
