#!/usr/bin/env Rscript
library(tidyverse)
library(biogram)
library(ranger)
library(tidyr)
library(stringi)
library(data.table)
#--------------------------------------------------FUNCTIONS

#----get_mers

create_mer_df <- function(seq) 
  do.call(rbind, lapply(1L:nrow(seq), function(i) {
    seq2ngrams(seq[i, ][!is.na(seq[i, ])], 10, a()[-1]) %>% 
      decode_ngrams() %>% 
      unname() %>% 
      strsplit(split = "") %>% 
      do.call(rbind, .) %>% 
      data.frame(stringsAsFactors = FALSE) %>% 
      mutate(source_peptide = rownames(seq)[i],
             mer_id = paste0(source_peptide, "m", 1L:nrow(.)))
  }))


#--------------------- peptydy (count ampgrams)

count_ampgrams <- function(mer_df, ns, ds) {
  
  mer_df[, grep("^X", colnames(mer_df))] %>% 
    as.matrix() %>% 
    count_multigrams(ns = ns, 
                     ds = ds,
                     seq = .,
                     u = toupper(colnames(aaprop))) %>% 
    binarize
}
#---------------------------------------------------------------ANALYZIS
count_longest <- function(x) {
  splitted_x <- strsplit(x = paste0(as.numeric(x > 0.5), collapse = ""),
                         split = "0")[[1]]
  len <- unname(sapply(splitted_x, nchar))
  if (length(len[len > 0]) == 0) {
    0 } else {
      len[len > 0]
    }
}
#-----------------------------------------
calculate_statistics <- function(pred_mers) {
  (if("fold" %in% colnames(pred_mers)) {
    group_by(pred_mers, source_peptide, target, fold)
  } else {  
    group_by(pred_mers, source_peptide, target)
  }) %>% 
    summarise(fraction_true = mean(pred > 0.5),
              pred_mean = mean(pred),
              pred_median = median(pred),
              n_peptide = length(pred),
              n_pos = sum(pred > 0.5),
              pred_min = min(pred),
              pred_max = max(pred), 
              longest_pos = max(count_longest(pred)),
              n_pos_10 = sum(count_longest(pred) >= 10),
              frac_0_0.2 = sum(pred <= 0.2)/n(),
              frac_0.2_0.4 = sum(pred > 0.2 & pred <= 0.4)/n(),
              frac_0.4_0.6 = sum(pred > 0.4 & pred <= 0.6)/n(),
              frac_0.6_0.8 = sum(pred > 0.6 & pred <= 0.8)/n(),
              frac_0.8_1 = sum(pred > 0.8 & pred <= 1)/n()) %>% 
    ungroup() %>% 
    mutate(target = factor(target))
}

train_model_peptides <- function(mer_statistics) {
  train_dat <- mer_statistics %>% 
    select(c("target", "fraction_true", "pred_mean", "pred_median",
             "n_peptide", "n_pos", "pred_min", "pred_max", "longest_pos",
             "n_pos_10", "frac_0_0.2", "frac_0.2_0.4", "frac_0.4_0.6",
             "frac_0.6_0.8", "frac_0.8_1"))
  model_cv <- ranger(dependent.variable.name = "target", data = train_dat, 
                     write.forest = TRUE, probability = TRUE, num.trees = 500, 
                     verbose = FALSE, classification = TRUE, seed = 990)
  model_cv
}

#------

args = commandArgs(trailingOnly=TRUE)

train_file <- args[1]
test_file <- args[2]
output_file <- args[3]

training_data <- read_fasta(train_file) %>% 
  list2matrix() %>% 
  create_mer_df()

ngrams12 <- count_ampgrams(training_data, ns = c(1, rep(2, 4)),ds = list(0, 0, 1, 2, 3))
ngrams3_1 <-  count_ampgrams(training_data, ns = c(3, 3), ds = list(c(0, 0), c(0, 1)))
ngrams3_2 <-  count_ampgrams(training_data, ns = c(3, 3), ds = list(c(1, 0), c(1, 1)))

binary_ngrams <-  cbind(ngrams12,ngrams3_1,ngrams3_2)

train_dat <- training_data %>% 
  mutate(target = as.numeric((stringi::stri_match_last_regex(source_peptide, "(?<=AMP\\=)0|1") == "1"))) %>% 
  mutate(target = ifelse(target == 1,1,0))


test_bis <- test_features(train_dat[["target"]],binary_ngrams)

imp_bigrams <- cut(test_bis, breaks = c(0, 0.05, 1))[[1]]

ranger_train_data <- data.frame(as.matrix(binary_ngrams[, imp_bigrams]),
                                tar = as.factor(train_dat[["target"]]))

model_cv <- ranger(dependent.variable.name = "tar", data = ranger_train_data, 
                   write.forest = TRUE, probability = TRUE, num.trees = 2000, 
                   verbose = FALSE)

pred <-  predict(model_cv, data.frame(as.matrix(binary_ngrams[, imp_bigrams])))

mer_df <- train_dat %>% 
  mutate(pred=pred[["predictions"]][, "1"]) %>% 
  select(c("source_peptide","mer_id","target", "pred"))

mer_statistics <- calculate_statistics(mer_df)

peptydovy <- train_model_peptides(mer_statistics)


#------------------------test

test_data <- read_fasta(test_file) %>% 
  list2matrix() %>% 
  create_mer_df()

ngrams12_T <- count_ampgrams(test_data, ns = c(1, rep(2, 4)),ds = list(0, 0, 1, 2, 3))
ngrams3_1_T <-  count_ampgrams(test_data, ns = c(3, 3), ds = list(c(0, 0), c(0, 1)))
ngrams3_2_T <-  count_ampgrams(test_data, ns = c(3, 3), ds = list(c(1, 0), c(1, 1)))


binary_ngrams_T <-  cbind(ngrams12_T,ngrams3_1_T,ngrams3_2_T)

test_dat <- test_data %>% 
  mutate(target = as.numeric((stringi::stri_match_last_regex(source_peptide, "(?<=AMP\\=)0|1") == "1"))) %>% 
  mutate(target = ifelse(target == 1,1,0))


pred_T <- predict(model_cv, data.frame(as.matrix(binary_ngrams_T[, imp_bigrams])))

mer_df_T <- test_dat %>% 
  mutate(pred=pred_T[["predictions"]][, "1"]) %>% 
  select(c("source_peptide","mer_id","target", "pred"))

mer_statistics_T <- calculate_statistics(mer_df_T)



gg <- predict(peptydovy, mer_statistics_T) 

getpred <- as.data.frame(gg$predictions)

mer_statistics_T %>% 
  select(c("source_peptide","target")) %>% 
  mutate(probability=getpred[["1"]]) %>% 
  rename("ID"="source_peptide") %>% 
  mutate(prediction=ifelse(probability > 0.5, 1, 0)) %>% 
  write.csv(file=output_file, row.names = FALSE, quote = FALSE)


