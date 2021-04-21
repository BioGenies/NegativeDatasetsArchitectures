#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(rBLAST)
library(dplyr)
library(stringr)
library(e1071)

read_fasta <- function(file) {
  all_lines <- readLines(file)
  prot_id <- cumsum(grepl("^>", all_lines))
  all_prots <- split(all_lines, prot_id)
  
  seq_list <- lapply(all_prots, function(ith_seq)
    unlist(strsplit(ith_seq[-1], split = "")))
  
  names(seq_list) <- sub(">", "", sapply(all_prots, function(ith_seq) ith_seq[1]), fixed = TRUE)
  
  seq_list
}

create_target <- function(fasta_file) {
  target <- unlist(lapply(fasta_file, function(x) ifelse(grepl("AMP=1", attr(x, "name")) ,1 , 0)))
  names(target) <- NULL
  target <- as.factor(target)
  
  target
}

# save sequences to a format acceptable by perl script (LZ complexity features)
save_to_LZ_format <- function(sequences, filename) {
  text <- paste0(lapply(sequences, function(x)
    paste0("\n", paste0(x, collapse=""), collapse="")), collapse="")
  
  writeLines(text, filename)
}

# parse perl script output
parse_LZ_features <- function(filepath) {
  
  lines <- readLines(filepath)
  
  x <- lapply(lines, function(line) {
    as.numeric(unlist(lapply(str_split((str_split(line, pattern = "\t")[[1]][2:1001]), pattern=":"), 
                             function(x) {x[2]}
    )
    )
    )
  })
  
  do.call(rbind, x)
}

## SETUP

train_path <- args[1]
test_path <- args[2]
output_path <- args[3]

metric_column = "Perc.Ident"

## Read data

train <- read_fasta(train_path)
test <- read_fasta(test_path)

y_train <- create_target(train)
y_test <- create_target(test)



## 1000 sequences to create LZ features

trainPos <- train[y_train == 1]
trainNeg <- train[y_train == 0]

numSeqLZ <- 500

LZindices_train <- sample(1:length(trainPos), size = numSeqLZ, replace = FALSE)
LZindices_test <- sample(1:length(trainNeg), size = numSeqLZ, replace = FALSE)

LZ_sequences_train_pos <- trainPos[LZindices_train]
train_sequences_pos <- trainPos[-LZindices_train]

LZ_sequences_train_neg <- trainNeg[LZindices_test]
train_sequences_neg <- trainNeg[-LZindices_test]

train_reduced <- c(train_sequences_pos, train_sequences_neg)
y_train_reduced <- create_target(train_reduced)

LZ_sequences <- c(LZ_sequences_train_pos, LZ_sequences_train_neg)

## create output dataframes

train_df <- data.frame(id = names(train_reduced), pred =NA)
test_df <- data.frame(id = names(test), pred =NA)

## BLASTP predictions

blastDir <- "./tmp-blast-dir/"
fasta_name <- "tmp-db-sequences.fasta"
dir.create(blastDir)
# Positive sequences BLAST database
write.fasta(train_reduced, names(train), paste0(blastDir, fasta_name))
makeblastdb(paste0(blastDir, fasta_name), dbtype = "prot")

# predict on both train and test sequences

blastClf <- blast(db=paste0(blastDir, fasta_name), type = "blastp")

trainAAStringSet <- readAAStringSet(paste0(blastDir, fasta_name))
testAAStringSet <- readAAStringSet(test_path)

BLASTP_predictions_train <- predict(blastClf, trainAAStringSet)
BLASTP_predictions_test <- predict(blastClf, testAAStringSet)

# remove alignments to sequences itself
BLASTP_predictions_train <- BLASTP_predictions_train[BLASTP_predictions_train$Perc.Ident != 100, ]


## AMP predictions based on alignments for both train and test sequences

BLAST_preds_train <- lapply(unique(BLASTP_predictions_train[["QueryID"]]), function(seq_name) {
  
  blast_pred <- BLASTP_predictions_train[BLASTP_predictions_train[["QueryID"]] == seq_name, ]
  best_subject <- blast_pred[which.max(blast_pred[[metric_column]]), "SubjectID"]
  ifelse(grepl("AMP=1", best_subject), 1, 0)
  
})

BLAST_preds_test <- lapply(unique(BLASTP_predictions_test[["QueryID"]]), function(seq_name) {
  
  blast_pred <- BLASTP_predictions_test[BLASTP_predictions_test[["QueryID"]] == seq_name, ]
  best_subject <- blast_pred[which.max(blast_pred[[metric_column]]), "SubjectID"]
  ifelse(grepl("AMP=1", best_subject), 1, 0)
  
})

names(BLAST_preds_train) <- unique(BLASTP_predictions_train[["QueryID"]])
names(BLAST_preds_test) <- unique(BLASTP_predictions_test[["QueryID"]])

for (seqname in names(BLAST_preds_train)) {
  train_df[train_df$id == seqname, "pred"] <- BLAST_preds_train[[seqname]]
}

for (seqname in names(BLAST_preds_test)) {
  test_df[test_df$id == seqname, "pred"] <- BLAST_preds_test[[seqname]]
}

## Sequences w/o alignment hit will be used in SVM-LZ
train_sequences_wo_hit <- train_reduced[train_df[is.na(train_df$pred), "id"]]
test_sequences_wo_hit <- test[test_df[is.na(test_df$pred), "id"]]

lapply(train_sequences_wo_hit, function(x) grepl("AMP=1", attr(x, "name")))

posIndices <- unlist(lapply(train_sequences_wo_hit, function(x) grepl("AMP=1", attr(x, "name"))))

train_sequences_wo_hit_pos <- train_sequences_wo_hit[posIndices]
train_sequences_wo_hit_neg <- train_sequences_wo_hit[-posIndices]

# save file so that they can be used in a perl script

save_to_LZ_format(LZ_sequences, "Fixed_AMPtrainingset.txt")
save_to_LZ_format(test_sequences_wo_hit, "0New_AMPtest.txt")
save_to_LZ_format(train_sequences_wo_hit_pos, "0New_AMPtrain+.txt")
save_to_LZ_format(train_sequences_wo_hit_neg, "0New_AMPtrain-.txt")

system("perl models/SVM-LZ/LZ.pl 0")
system("perl models/SVM-LZ/LZ.pl 1")
system("perl models/SVM-LZ/LZ.pl 2")

## parse LZ features
library(stringr)

X_train_pos <- parse_LZ_features("./FV_0New_AMPtrain+.txt")
X_train_neg <- parse_LZ_features("./FV_0New_AMPtrain-.txt")
X_train <- rbind(X_train_pos, X_train_neg)

X_test <- parse_LZ_features("./FV_0New_AMPtest.txt")

y_train <- c(rep(1, nrow(X_train_pos)), rep(0, nrow(X_train_neg)))
y_train <- factor(y_train, levels=c(1, 0))

# radial basis function (RBF) kernel
# C = 4, 𝛾 = 3.91e-3 (normalny training dataset)
svm_model <- svm(X_train, y_train, kernel = "radial", cost = 4)
svm_predictions <- predict(svm_model, X_test)
svm_predictions <- as.numeric(as.character(svm_predictions))

names(svm_predictions) <- names(test_sequences_wo_hit)

for (seqname in names(svm_predictions)) {
  test_df[test_df$id == seqname, "pred"] <- svm_predictions[[seqname]]
}

write.csv(test_df, file=output_path, row.names=FALSE)

file.remove("Fixed_AMPtrainingset.txt")
file.remove("0New_AMPtest.txt")
file.remove("0New_AMPtrain+.txt")
file.remove("0New_AMPtrain-.txt")
file.remove("FV_0New_AMPtest.txt")
file.remove("FV_0New_AMPtrain+.txt")
file.remove("FV_0New_AMPtrain-.txt")


