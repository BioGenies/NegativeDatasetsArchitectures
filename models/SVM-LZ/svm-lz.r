library(rBLAST)
library(seqinr)
library(dplyr)

create_target <- function(fasta_file) {
  target <- unlist(lapply(fasta_file, function(x) ifelse(grepl("AMP=1", attr(x, "name")) ,1 , 0)))
  names(target) <- NULL
  target <- as.factor(target)
  
  target
}

train_path <- "./example-data/exemplary_train_dataset.fa"
test_path <- "./example-data/exemplary_test_dataset.fa"

train <- read.fasta(train_path, seqtype="AA")
test <- read.fasta(test_path, seqtype="AA")

y_train <- create_target(train)
y_test <- create_target(test)

# subset sequences to LZ phase
trainPos <- train[y_train == 1]
trainNeg <- train[y_train == 0]

numSeqLZ <- 500

LZindices_train <- sample(1:length(trainPos), size = numSeqLZ, replace = FALSE)
LZindices_test <- sample(1:length(trainNeg), size = numSeqLZ, replace = FALSE)

LZ_sequences_train_pos <- trainPos[LZindices_train]
train_sequences_pos <- trainPos[-LZindices_train]

LZ_sequences_train_neg <- trainNeg[LZindices_test]
train_sequences_neg <- trainNeg[-LZindices_test]


train <- c(train_sequences_pos, train_sequences_neg)
y_train <- create_target(train)

# BLAST
blastDir <- "./tmp-blast-dir/"
fasta_name <- "tmp-db-sequences.fasta"
dir.create(blastDir)
# Positive sequences BLAST database
setwd(blastDir)
write.fasta(train, names(train), fasta_name)
makeblastdb(fasta_name, dbtype = "prot")
setwd("..")

# predict on test sequence

blastClf <- blast(db=paste0(blastDir, fasta_name), type = "blastp")

testAAStringSet <- readAAStringSet(test_path)

BLASTP_predictions <- predict(blastClf, testAAStringSet)

##########################
example_seq_name <- names(testAAStringSet)[1]

blastp_pred_example <- BLASTP_predictions[BLASTP_predictions$QueryID == example_seq_name,]

metric_column = "Perc.Ident"
best_subject <- blastp_pred_example[which.max(blastp_pred_example[[metric_column]]), "SubjectID"]

ifelse(grepl("AMP=1", best_subject), 1, 0)

BLAST_preds <- lapply(unique(BLASTP_predictions[["QueryID"]]), function(seq_name) {
  
  blast_pred <- BLASTP_predictions[BLASTP_predictions[["QueryID"]] == seq_name, ]
  best_subject <- blast_pred[which.max(blast_pred[[metric_column]]), "SubjectID"]
  ifelse(grepl("AMP=1", best_subject), 1, 0)
  
})

names(BLAST_preds) <- unique(BLASTP_predictions[["QueryID"]])

for (seq_name in names(test)){
  print(seq_name)
}

#TODO: LZ complexity 
