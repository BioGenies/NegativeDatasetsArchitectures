#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(e1071)
library(seqinr)
source("./models/CS-AMPPred/features.R")

runCSAMPPred <- function(train_path, test_path, output_path) {
  
  train_features <- create_features(train_path)
  test_features <- create_features(test_path)
  
  train_target <- create_target(train_path)
  test_target <- create_target(test_path)
  
  pca <- prcomp(train_features)
  
  train_pca <- predict(pca, train_features)
  test_pca <- predict(pca, test_features)
  
  pvals <- c()
  for (i in 1:ncol(train_pca)) {
    p_val <- wilcox.test(train_pca[train_target == 1, i], train_pca[train_target == 0, i], alternative="two.sided")$p.value
    pvals <- c(pvals, p_val)
  }
  
  X_train <- train_pca[, pvals < 0.05]
  X_test <- test_pca[, pvals < 0.05]
  
  # which model is appropiate?
  # AFAIK: no probability, only classes predicted
  svm_linear <- svm(X_train, train_target, kernel = "linear")
  # svm_radial <- svm(X_train, train_target, kernel = "radial")
  # svm_poly <- svm(X_train, train_target, kernel = "polynomial")
  
  predictions <- as.character(predict(svm_linear, X_test))
  y_test <- as.character(test_target)
  test_seqnames <-rownames(test_features)
  
  out <- data.frame(cbind(test_seqnames, y_test, predictions))
  colnames(out) <- c("ID", "target", "prediction")
  
  write.csv(out, file = output_path, row.names = FALSE, quote = FALSE)
  
  TRUE
}

runCSAMPPred(args[1], args[2], args[3])

