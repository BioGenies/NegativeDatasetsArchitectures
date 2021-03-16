# MACREL features adopted from
# https://github.com/BigDataBiology/macrel/blob/master/macrel/AMP_features.py

import argparse
import re

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects
from rpy2.robjects import numpy2ri
from sklearn.ensemble import RandomForestClassifier

numpy2ri.activate()
r = rpy2.robjects.r
r.library('Peptides')


def fasta_iter(fname, full_header=False):
    """Iterate over a (possibly gzipped) FASTA file

    Parameters
    ----------
    fname : str
        Filename. If it ends with .gz, gzip format is assumed
    full_header : boolean (optional)
        If True, yields the full header. Otherwise (the default), only the
        first word

    Yields
    ------
    (h,seq, class): tuple of (str, str, str)
    """
    header = None
    chunks = []
    if fname.endswith('.gz'):
        import gzip
        op = gzip.open
    else:
        op = open
    with op(fname, 'rt') as f:
        for line in f:
            if line[0] == '>':
                target = re.search("AMP=(.*)", line).group()[-1]
                if header is not None:
                    yield header, ''.join(chunks), target
                line = line[1:].strip()
                if not line:
                    header = ''
                elif full_header:
                    header = line.strip()
                else:
                    header = line.split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, ''.join(chunks), target


GROUPS_SA = ['ALFCGIVW', 'RKQEND', 'MSPTHY']  # solventaccess
GROUPS_HB = ['ILVWAMGT', 'FYSQCN', 'PHKEDR']  # HEIJNE&BLOMBERG1979

# ' # http://emboss.bioinformatics.nl/cgi-bin/emboss/pepstats
# ' # Property      Residues              Number  Mole%
# ' # Tiny          (A+C+G+S+T)             4   19.048
# ' # Small         (A+B+C+D+G+N+P+S+T+V)   4   19.048
# ' # Aliphatic     (A+I+L+V)               5   23.810
# ' # Aromatic      (F+H+W+Y)               5   23.810
# ' # Non-polar     (A+C+F+G+I+L+M+P+V+W+Y) 11  52.381
# ' # Polar         (D+E+H+K+N+Q+R+S+T+Z)   9   42.857
# ' # Charged       (B+D+E+H+K+R+Z)         8   38.095
# ' # Basic         (H+K+R)                 8   38.095
# ' # Acidic        (B+D+E+Z)               0   00.000

_aa_groups = [
    set('ACGST'),  # Tiny
    set('ABCDGNPSTV'),  # Small
    set('AILV'),  # Aliphatic
    set('FHWY'),  # Aromatic
    set('ACFGILMPVWY'),  # Nonpolar
    set('DEHKNQRSTZ'),  # Polar
    set('BDEHKRZ'),  # Charged
    set('HKR'),  # Basic
    set('BDEZ'),  # Acidic
]


def amino_acid_composition(seq):
    """amino_acid_composition: return AA composition fractions"""
    # See groups above
    return np.array(
        [sum(map(g.__contains__, seq)) for g in _aa_groups],
        dtype=float) / len(seq)


def ctdd(sequence, groups):
    code = []
    for group in groups:
        for i, aa in enumerate(sequence):
            if aa in group:
                code.append((i + 1) / len(sequence) * 100)
                break
        else:
            code.append(0)
    return code


def features(ifile):
    groups = [set(g) for g in (GROUPS_SA + GROUPS_HB)]
    targets = []
    seqs = []
    headers = []
    encodings = []
    aaComp = []
    for h, seq, target in fasta_iter(ifile):
        if seq[-1] == '*':
            seq = seq[:-1]
        if seq[0] == 'M':
            seq = seq[1:]
        seqs.append(seq)
        headers.append(h)
        targets.append(int(target))
        encodings.append(ctdd(seq, groups))
        aaComp.append(amino_acid_composition(seq))

    # We can do this inside the loop so that we are not forced to pre-load all
    # the sequences into memory. However, it becomes much slower
    rpy2.robjects.globalenv['seq'] = seqs
    rfeatures = r('''
    ch <- charge(seq=seq, pH=7, pKscale="EMBOSS")
    pI <- pI(seq=seq, pKscale="EMBOSS")
    aIndex <- aIndex(seq=seq)
    instaIndex <- instaIndex(seq=seq)
    boman <- boman(seq=seq)
    hydrophobicity <- hydrophobicity(seq=seq, scale="Eisenberg")
    hmoment <- hmoment(seq=seq, angle=100, window=11)
    cbind(ch, pI, aIndex, instaIndex, boman, hydrophobicity, hmoment)
    ''')

    features = np.hstack([aaComp, rfeatures, encodings])

    # This is arguably a Pandas bug (at least inconsistency), but
    # pd.DataFrame([], ...) works, while pd.DataFrame(np.array([]), ...) does
    # not:
    if len(features) == 0:
        features = []

    features = pd.DataFrame(features, index=headers, columns=[
        "tinyAA",
        "smallAA",
        "aliphaticAA",
        "aromaticAA",
        "nonpolarAA",
        "polarAA",
        "chargedAA",
        "basicAA",
        "acidicAA",
        "charge",
        "pI",
        "aindex",
        "instaindex",
        "boman",
        "hydrophobicity",
        "hmoment",
        "SA.Group1.residue0",
        "SA.Group2.residue0",
        "SA.Group3.residue0",
        "HB.Group1.residue0",
        "HB.Group2.residue0",
        "HB.Group3.residue0",
    ])
    features.insert(0, 'group', 'Unk')
    features.insert(0, 'target', targets)
    features.insert(0, 'sequence', seqs)
    return features


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", "-t", help="path to training set")
    parser.add_argument("--test", "-e", help="path to test set")
    parser.add_argument("--output", "-o", help="path to save results")

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    train_features = features(args.train)
    test_features = features(args.test)

    rf_wout_oob = RandomForestClassifier(
        n_estimators=101,
        random_state=12345,
        n_jobs=8)
    rf_wout_oob.fit(train_features.iloc[:, 3:], train_features['target'])

    results = pd.DataFrame({
        'id': test_features.index,
        'target': test_features.target,
        'predict': rf_wout_oob.predict_proba(test_features.iloc[:, 3:])[:, 1]},
    ).reset_index(drop=True)

    results.to_csv(args.output, index=False)

    return


if __name__ == "__main__":
    main()
