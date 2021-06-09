from __future__ import print_function # enable Python3 printing
import re
import argparse
import numpy
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.utils import shuffle
from Bio import SeqIO
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

            
def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", "-t", help="path to training set")
    parser.add_argument("--test", "-e", help="path to test set")
    parser.add_argument("--output", "-o", help="path to save results")

    args = parser.parse_args()

    return args 
    
    
def main():
    
    args = parse_arguments()
    
    train_file = args.train
    test_file = args.test

    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    aa2int = dict((c, i) for i, c in enumerate(amino_acids))

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    ids_test = []

    for s in SeqIO.parse(train_file,"fasta"):
        X_train.append([aa2int[aa] for aa in str(s.seq).upper()])
        if re.search("AMP=1", s.id):
            y_train.append(1)
        elif re.search("AMP=0", s.id):
            y_train.append(0)
    	
    for s in SeqIO.parse(test_file,"fasta"):
        X_test.append([aa2int[aa] for aa in str(s.seq).upper()])
        ids_test.append(s.id)
        if re.search("AMP=1", s.id):
            y_test.append(1)
        elif re.search("AMP=0", s.id):
            y_test.append(0)
     
    max_length = 500
    embedding_vector_length = 128
    nbf = 64 		# No. Conv Filters
    flen = 16 		# Conv Filter length 
    nlstm = 100 	# No. LSTM layers
    ndrop = 0.1     # LSTM layer dropout
    nbatch = 32 	# Fit batch No.
    nepochs = 10    # No. training rounds

    # Pad input sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_length)

    # Shuffle training sequences
    X_train, y_train = shuffle(X_train, numpy.array(y_train))

    model = Sequential()
    model.add(Embedding(21, embedding_vector_length, input_length=max_length))
    model.add(Conv1D(filters=nbf, kernel_size=flen, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(LSTM(nlstm, use_bias=True, dropout=ndrop, return_sequences=False))#,merge_mode='ave'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, numpy.array(y_train), epochs=nepochs, batch_size=nbatch, verbose=1)

    preds = model.predict(X_test)
    pred_class = numpy.rint(preds) #round up or down at 0.5
	
    results = pd.DataFrame({
        'ID': ids_test,
        'target': y_test,
        'prediction': pred_class.astype(numpy.int32).flatten().tolist(),
        'probability': preds.flatten().tolist()},
    ).reset_index(drop=True)
    
    results.to_csv(args.output, index=False)
    
if __name__=="__main__":
    main()
