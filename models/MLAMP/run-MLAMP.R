#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(biogram)
library(dplyr)
library(protr)
library(RWeka)

#' Create grey model GM(1, 1)
#' 
#' @description Builds grey model with parameters 1 and 1
#' from the supplied data. Returns a numeric vector of
#' length 2 with \code{a} and \code{b} coefficients for 
#' supplied observation.
#' 
#' @param row
#'  Training data in numeric vector format without target
#'  value. Should contain data about one sequence.
#' 
#' @examples
#' X <- list(
#'   X1 = 1:5,
#'   X2 = c(10, 12, 24, 60),
#'   X3 = 9:2
#' )
#' lapply(X, grey_model_1_1)
#' 
#' @noRd
grey_model_1_1 <- function(row) {
  # First we compute first-order AGO
  ago <- cumsum(row)
  # Unfortunately, had to write for-loop, because of trying to
  #  iterate over rows of two matrices simultaneously.
  # -a is called "developing coefficient"
  # b is called "influence coefficient"
  # [a, b]^T = [B^T*B]^-1*B^T*Y
  B <- cbind(-0.5*(ago[-length(ago)] + ago[-1]), 1)
  Y <- row[-1]
  # ^ has higher precedence than %*%, so no parentheses necessary
  ret <- as.numeric((t(B) %*% B)^-1 %*% t(B) %*% Y)
  setNames(ret, c("a", "b"))
}

my_props <- data.frame(
  AccNo = c("hydrophobicity", "pK1", "pK2", "pI", "mw"),
  A = c(0.62, 2.35, 9.87, 6.11, 15),
  R = c(-2.53, 2.18, 9.09, 10.76, 101),
  N = c(-0.78, 2.18, 9.09, 10.76, 58), 
  D = c(-0.9, 1.88, 9.6, 2.98, 59), 
  C = c(0.29, 1.71, 10.78, 5.02, 47), 
  E = c(-0.74, 2.19, 9.67, 3.08, 73), 
  Q = c(-0.85, 2.17, 9.13, 5.65, 72), 
  G = c(0.48, 2.34, 9.6, 6.06, 1), 
  H = c(-0.4, 1.78, 8.97, 7.64, 82), 
  I = c(1.38, 2.32, 9.76, 6.04, 57), 
  L = c(1.06, 2.36, 9.6, 6.04, 57), 
  K = c(-1.5, 2.2, 8.9, 9.47, 73), 
  M = c(0.64, 2.28, 9.21, 5.74, 75), 
  F = c(1.19, 2.58, 9.24, 5.91, 91), 
  P = c(0.12, 1.99, 10.6, 6.3, 42), 
  S = c(-0.18, 2.21, 9.15, 5.68, 31), 
  T = c(-0.05, 2.15, 9.12, 5.6, 45), 
  W = c(0.81, 2.38, 9.39, 5.88, 130), 
  Y = c(0.26, 2.2, 9.11, 5.63, 107), 
  V = c(1.08, 2.29, 9.74, 6.02, 43)
)

generate_features <- function(sequence_file) {
  seqs <- read_fasta(sequence_file)
  # 20 features - amino acid composition
  aa_comp <- lapply(1:length(seqs), function(i) {
    data.frame(id = names(seqs[i]),
               t(as.matrix(table(seqs[[i]])/20)))
  }) %>% bind_rows()
  aa_comp[is.na(aa_comp)] <- 0
  # 10 features - coefficients of grey model when amino acids are encoded 
  # by numerical values of different properties
  gm_coefficients <- lapply(my_props[["AccNo"]], function(ith_prop) {
    lapply(1:length(seqs), function(i) {
      coeffs <- extractPAAC(x = paste(seqs[[i]], collapse = ""),
                            props = ith_prop,
                            customprops = my_props,
                            w = 0.1,
                            lambda = 4) %>% 
        grey_model_1_1()
      data.frame(a = -coeffs[["a"]],
                 b = coeffs[["b"]]) %>% 
        setNames(c(paste0(ith_prop, "_-a"), paste0(ith_prop, "_b")))
    }) %>% bind_rows()
  }) %>% bind_cols()
  bind_cols(aa_comp, gm_coefficients) %>% 
    mutate(target = as.factor(ifelse(grepl("AMP=1", id), 1, 0)))
}


train_df <- generate_features(args[1])
test_df <- generate_features(args[2])

rf <- make_Weka_classifier("weka/classifiers/trees/RandomForest")
trained_rf <- rf(formula = target ~ .,
                 data = train_df[, 2:ncol(train_df)],
                 control = Weka_control(I = 500, K = 6))

preds <- predict(trained_rf, 
                 select(test_df, -c("id", "target")),
                 type = "probability") %>% 
  as.data.frame() %>% 
  mutate(prediction = ifelse(`1` > 0.5, 1, 0)) 

res <- bind_cols(test_df[, c("id", "target")],
                 preds[, c("prediction", "1")]) 
colnames(res) <- c("id", "target", "prediction", "probability")

write.csv(res, args[3], row.names = FALSE)
