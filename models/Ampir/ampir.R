#!/usr/bin/env Rscript
library(ampir)
library(caret)
library(tidyverse)

args = commandArgs(trailingOnly=TRUE)

train_file <- args[1]
test_file <- args[2]
output_file <- args[3]

used_predictors <- c("Amphiphilicity", "Hydrophobicity", "pI", "Mw", 
                     "Charge", "Xc1.A", "Xc1.R", "Xc1.N", "Xc1.D", "Xc1.C", "Xc1.E", 
                     "Xc1.Q", "Xc1.G", "Xc1.H", "Xc1.I", "Xc1.L", "Xc1.K", "Xc1.M", 
                     "Xc1.F", "Xc1.P", "Xc1.S", "Xc1.T", "Xc1.W", "Xc1.Y", "Xc1.V", 
                     "Xc2.lambda.1", "Xc2.lambda.2")

ampir_train_df <- read_faa(train_file) %>%
  mutate(Label = as.numeric((stringi::stri_match_last_regex(seq_name, "(?<=AMP\\=)0|1") == "1"))) %>%
  mutate(Label = ifelse(Label == 1,"Positive","Negative"))  %>% remove_nonstandard_aa()


ampir_features <- calculate_features(ampir_train_df, min_len = 5)
ampir_features$Label <- as.factor(ampir_train_df$Label)
rownames(ampir_features) <- NULL


trctrl_prob <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                            classProbs = TRUE)

ampir_svm_model <- train(Label~.,
                         data = select(ampir_features, c(used_predictors, "Label")), 
                         method="svmRadial",
                         trControl = trctrl_prob,
                         preProcess = c("center", "scale"))


#---- prediction
ampir_test_df <- read_faa(test_file) %>%
  mutate(Label = as.numeric((stringi::stri_match_last_regex(seq_name, "(?<=AMP\\=)0|1") == "1"))) 

ampir_AMPs <- predict_amps(ampir_test_df, min_len = 5, model = ampir_svm_model)

data.frame(ID = ampir_test_df[["seq_name"]],
           target = ampir_test_df[["Label"]],
           probability = ampir_AMPs[["prob_AMP"]]) %>% 
  mutate(prediction = ifelse(probability > 0.5, 1, 0)) %>%
  write.csv(file = output_file, row.names = FALSE)
