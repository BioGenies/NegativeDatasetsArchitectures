from Bio import SeqIO
from Bio.Seq import Seq
from string import ascii_uppercase
import numpy as np

positive = []

with open("/home/jakubkala/Downloads/uniprot-anti+microbial-filtered-organism__Homo+sapiens+(Human)+[9606%5--.fasta", "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        record.description += " AMP=1"
        positive.append(record)

seq_lengths = [len(i.seq) for i in positive]

negative = []
for i in range(99):
    name = f"sp|example|negative_{i}" 
    seq = "".join(np.random.choice(list(ascii_uppercase), replace=True, size=np.random.choice(seq_lengths)))
    negative.append(SeqIO.SeqRecord(id= name, seq = Seq(seq), name = name, description=f"{name} AMP=0"))

with open("example-train.fasta", "w") as output_handle:
    SeqIO.write(positive+negative, output_handle, "fasta")