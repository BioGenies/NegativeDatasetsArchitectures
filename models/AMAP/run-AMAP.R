#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(dplyr)
library(stringi)
library(protr)
library(e1071)
if (!require(tidysq)) devtools::install_github("BioGenies/tidysq")
if (!require(fknn)) devtools::install_github("DominikRafacz/fknn")

# the final model used in article is SVM on AAC features
prepare_features <- function(input_file) {
  sequences_df <- read_fasta(input_file, alphabet = "ami_bsc")
  
  sequences_df %>%
    pull(sq) %>%
    sqapply(protr::extractAAC,
            single_string = TRUE) %>%
    do.call(rbind, .) %>%
    as_tibble() %>%
    bind_cols(sequences_df, .) %>%
    mutate(label = as.factor((stringi::stri_match_last_regex(name, "(?<=AMP\\=)0|1") == "1")))
}

model <- prepare_features(args[1]) %>%
  select(-name, -sq) %>%
  e1071::svm(label~., .) # default parameters used as article does not mention anything in this topic

test_data <- prepare_features(args[2])

prediction <- predict(model, newdata = test_data %>% select(-name, -sq, -label)) 

test_data %>%
  select(ID = name,
         target = label) %>%
  mutate(target = as.numeric(target == "TRUE")) %>%
  bind_cols(., prediction = as.numeric(prediction == "TRUE")) %>%
  mutate(probability = NA_real_) %>%
  write.csv(args[3], row.names = FALSE)
