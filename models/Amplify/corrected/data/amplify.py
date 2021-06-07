#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from keras.models import load_model, Model
from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout, Layer
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from keras import backend as K
from keras import initializers, regularizers, constraints, activations
import re
import pandas as pd

class Attention(Layer):

    def __init__(self,
                 activation='tanh',
                 initializer='glorot_uniform',
                 return_attention=False,
                 W_regularizer=None,
                 u_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 u_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        self.supports_masking = True
        self.return_attention = return_attention

        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, time, amount_features)
        
        # the attention size matches the feature dimension
        amount_features = input_shape[-1]
        attention_size  = input_shape[-1]

        self.W = self.add_weight(shape=(amount_features, attention_size),
                                 initializer=self.initializer,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='attention_W')
        self.b = None
        if self.bias:
            self.b = self.add_weight(shape=(attention_size,),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     name='attention_b')

        self.context = self.add_weight(shape=(attention_size,),
                                       initializer=self.initializer,
                                       regularizer=self.u_regularizer,
                                       constraint=self.u_constraint,
                                       name='attention_us')

        super().build(input_shape)

    def call(self, x, mask=None):        
        # U = tanh(H*W + b) (eq. 8)        
        ui = K.dot(x, self.W)              # (b, t, a)
        if self.b is not None:
            ui += self.b
        ui = self.activation(ui)           # (b, t, a)

        # Z = U * us (eq. 9)
        us = K.expand_dims(self.context)   # (a, 1)
        ui_us = K.dot(ui, us)              # (b, t, a) * (a, 1) = (b, t, 1)
        ui_us = K.squeeze(ui_us, axis=-1)  # (b, t, 1) -> (b, t)
        
        # alpha = softmax(Z) (eq. 9)
        alpha = self._masked_softmax(ui_us, mask) # (b, t)
        alpha = K.expand_dims(alpha, axis=-1)     # (b, t, 1)
        
        if self.return_attention:
            return alpha
        else:
            # v = alpha_i * x_i (eq. 10)
            return K.sum(x * alpha, axis=1)
    
    def _masked_softmax(self, logits, mask):


        b = K.max(logits, axis=-1, keepdims=True)
        logits = logits - b

        exped = K.exp(logits)

        # ignoring masked inputs
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            exped *= mask

        partition = K.sum(exped, axis=-1, keepdims=True)


        partition = K.maximum(partition, K.epsilon())

        return exped / partition

    def compute_output_shape(self, input_shape):

        if self.return_attention:
            return input_shape[:-1]
        else:
            return input_shape[:-2] + input_shape[-1:]

    def compute_mask(self, x, input_mask=None):

        return None

    def get_config(self):
        config = {
            'activation': self.activation,
            'initializer': self.initializer,
            'return_attention': self.return_attention,

            'W_regularizer': initializers.serialize(self.W_regularizer),
            'u_regularizer': initializers.serialize(self.u_regularizer),
            'b_regularizer': initializers.serialize(self.b_regularizer),

            'W_constraint': constraints.serialize(self.W_constraint),
            'u_constraint': constraints.serialize(self.u_constraint),
            'b_constraint': constraints.serialize(self.b_constraint),
            
            'bias': self.bias
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

class ScaledDotProductAttention(Layer):

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):

        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(0, key_len), axis=0)
            upper = K.expand_dims(K.arange(0, query_len), axis=-1)
            e *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
        if mask is not None:
            e *= K.cast(K.expand_dims(mask, axis=-2), K.floatx())
        a = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())
        v = K.batch_dot(a, value)
        if self.return_attention:
            return [v, a]
        return v

    
class MultiHeadAttention(Layer):

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 return_multi_attention=False,
                 **kwargs):

        self.supports_masking = True
        self.head_num = head_num
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.history_only = history_only
        self.return_multi_attention = return_multi_attention

        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
            'return_multi_attention': self.return_multi_attention
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.return_multi_attention:
            if isinstance(input_shape, list):
                q, k, v = input_shape
                return (q[0], self.head_num, q[1], k[1])
            return (input_shape[0], self.head_num, input_shape[1], input_shape[1])
        else:
            if isinstance(input_shape, list):
                q, k, v = input_shape
                return q[:-1] + (v[-1],)
            return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))
    
    @staticmethod
    def _reshape_attention_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len = input_shape[0], input_shape[1]
        return K.reshape(x, (batch_size // head_num, head_num, seq_len, seq_len))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)            
        y,a = ScaledDotProductAttention(
            return_attention=True,
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        
        y = self._reshape_from_batches(y, self.head_num)
        a = self._reshape_attention_from_batches(a, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
      
        if self.return_multi_attention:
            return a
        return y
    
def one_hot_padding(seq_list,padding):
    """
    Generate features for aa sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences, 
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*20
        one_hot[aa[i]][i] = 1 
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*20]*(padding-len(seq_list[i]))
        feat_list.append(feat)   
    return(np.array(feat_list))

def build_amplify():
    """
    Build the complete model architecture
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(name='Attention')(hidden)
    prediction = Dense(1, activation='sigmoid', name='Output')(hidden)
    model = Model(inputs=inputs, outputs=prediction)
    return model

def ensemble(model_list, X):
    """
    Ensemble the list of models with processed input X, 
    Return results for ensemble and individual models
    """
    indv_pred = [] # list of predictions from each individual model
    for i in range(len(model_list)):
        indv_pred.append(model_list[i].predict(X).flatten())
    ens_pred = np.mean(np.array(indv_pred), axis=0)
    return ens_pred, np.array(indv_pred)

def proba_to_class_name(scores):
    """
    Turn prediction scores into real class names.
    If score > 0.5, label the sample as AMP; else non-AMP.
    Input: scores - scores predicted by the model, 1-d array.
    Output: an array of class labels.
    """
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append('AMP')
        else:
            classes.append('non-AMP')
    return np.array(classes)


def build_attention():
    """
    Build the model architecture for attention output
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(return_attention=True, name='Attention')(hidden)
    model = Model(inputs=inputs, outputs=hidden)
    return model

def load_multi_model(sciezka_modelu, architecture):
    """
    Load multiple models with the same architecture in one function.
    Input: list of saved model weights files.
    Output: list of loaded models.
    """
    model_list = []
    for i in range(len(sciezka_modelu)):
        model = architecture()
       # model.load_weights(sciezka_modelu[i], by_name=True)
        model_list.append(model)
    return model_list

MAX_LEN= 400
def build_model():
    """
    Build and compile the model.
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(name='Attention')(hidden)
    prediction = Dense(1, activation='sigmoid', name='Output')(hidden)
    model = Model(inputs=inputs, outputs=prediction)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #best
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def predict_by_class(scores):
    
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append(1)
        else:
            classes.append(0)
    return np.array(classes)



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", "-t", help="path to training set")
    parser.add_argument("--test", "-e", help="path to test set")
    parser.add_argument("--output", "-o", help="path to save results")

    args = parser.parse_args()
    
    train_file = args.train
    test_file = args.test
    out_file = args.output
    
        
    AMP_train = []
    non_AMP_train = []  
    for seq_record in SeqIO.parse(train_file, 'fasta'):
            if re.search("AMP=1", seq_record.description):
                AMP_train.append(str(seq_record.seq))
            elif re.search("AMP=0", seq_record.description):
                non_AMP_train.append(str(seq_record.seq))
                
    
        
        
        
            # sequences for training sets
    train_seq = AMP_train + non_AMP_train


        # set labels for training sequences
    y_train = np.array([1]*len(AMP_train) + [0]*len(non_AMP_train))

        
        # shuffle training set
    train = list(zip(train_seq, y_train))
    random.Random(123).shuffle(train)
    train_seq, y_train = zip(*train)
    train_seq = list(train_seq)
    y_train = np.array((y_train))

        
        # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_train = one_hot_padding(train_seq, MAX_LEN) 
        
    indv_pred_train = [] # list of predicted scores for individual models on the training set
    
    #--------- test
    AMP_test = []
    non_AMP_test = []
    ids_test = []
    peptide = []
    
    for seq_record in SeqIO.parse(test_file, 'fasta'):
            ids_test.append(seq_record.id)
            peptide.append(seq_record.seq)
            
            if re.search("AMP=1", seq_record.description):
                AMP_test.append(str(seq_record.seq))
            elif re.search("AMP=0", seq_record.description):
                non_AMP_test.append(str(seq_record.seq))
                   
    test_seq = AMP_test + non_AMP_test
    # set labels for test sequences
    y_test = np.array([1]*len(AMP_test) + [0]*len(non_AMP_test))
    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_test = one_hot_padding(test_seq, MAX_LEN)
    #indv_pred_test = [] # list of predicted scores for individual models on the test set  
    
    #------------------------------
    
    
    
    ensemble_number = 2 # number of training subsets for ensemble
    ensemble2 = StratifiedKFold(n_splits=ensemble_number, shuffle=True, random_state=50)
    save_file_num = 0
    
    model_list2 = []
    for tr_ens, te_ens in ensemble2.split(X_train, y_train):
            model = build_model()
            early_stopping = EarlyStopping(monitor='val_accuracy',  min_delta=0.001, patience=50, restore_best_weights=True)
            model.fit(X_train[tr_ens], np.array(y_train[tr_ens]), epochs=10, batch_size=32, 
                      validation_data=(X_train[te_ens], y_train[te_ens]), verbose=2, initial_epoch=0, callbacks=[early_stopping])
            temp_pred_train = model.predict(X_train).flatten() # predicted scores on the [whole] training set from the current model
            indv_pred_train.append(temp_pred_train)
            model_list2.append(model)
            
            #save_file_num = save_file_num + 1
            #save_dir = '/home/filip' + '/' + 'model' + '_' + str(save_file_num) + '.h5'
            #save_dir_wt = '/home/filip' + '/' + 'model' + '_weights_' + str(save_file_num) + '.h5'
            #model.save(save_dir) #save
            #model.save_weights(save_dir_wt) #save
    
            # training and validation accuracy for the current model
            temp_pred_class_train_curr = predict_by_class(model.predict(X_train[tr_ens]).flatten())
            temp_pred_class_val = predict_by_class(model.predict(X_train[te_ens]).flatten())
    
            print('*************************** current model ***************************')
            print('current train acc: ', accuracy_score(y_train[tr_ens], temp_pred_class_train_curr))
            print('current val acc: ', accuracy_score(y_train[te_ens], temp_pred_class_val))
 
           # preds = model.predict(X_test).flatten()
           # pred_class = np.rint(preds)

   
    
        
    models = [model_list2[i] for i in range(len(model_list2))]
    out_model = load_multi_model(models, build_amplify)

    
    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_seq = one_hot_padding(peptide, MAX_LEN)
    y_score, y_indv_list = ensemble(out_model, X_seq)
    y_class = proba_to_class_name(y_score)
    #preds = model.predict(X_test).flatten()


    #print(y_score)
    print(MAX_LEN)
    #print(models)
    
    results = pd.DataFrame({
            'ID': ids_test,
            'target': y_score,
            'prediction': y_class,
            'probability': model.predict(X_test).flatten(),},
        ).reset_index(drop=True)
        
    results.to_csv(out_file, index=False)
if __name__=="__main__":
    main()
