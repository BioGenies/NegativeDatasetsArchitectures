#!/usr/bin/env Rscript
library(ampir)
library(caret)
library(tidyverse)

args = commandArgs(trailingOnly=TRUE)

train_file <- args[1]
test_file <- args[2]
output_file <- args[3]

ampir_train_df <- read_faa(train_file) %>% 
  mutate(Label = as.numeric((stringi::stri_match_last_regex(seq_name, "(?<=AMP\\=)0|1") == "1"))) %>% 
  mutate(Label = ifelse(Label == 1,"Positive","Negative"))  %>% remove_nonstandard_aa()


ampir_features <- calculate_features(ampir_train_df)
ampir_features$Label <- as.factor(ampir_train_df$Label)
rownames(ampir_features) <- NULL


trctrl_prob <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                            classProbs = TRUE)

#trainIndex <-createDataPartition(y=ampir_features$Label, p=.7, list = FALSE)
#ampir_featuresTrain <-ampir_features[trainIndex,]

ampir_svm_model <- train(Label~.,
                         data = ampir_features[,-1], # excluding seq_name column
                         method="svmRadial",
                         trControl = trctrl_prob,
                         preProcess = c("center", "scale"),na.action = na.exclude)


#---- prediction
ampir_test_df <- read_faa(test_file) %>% 
  mutate(Label = as.numeric((stringi::stri_match_last_regex(seq_name, "(?<=AMP\\=)0|1") == "1"))) %>% 
  mutate(Label = ifelse(Label == 1,"Positive","Negative")) %>% remove_nonstandard_aa()

ampir_featuresTest <- calculate_features(ampir_test_df)
ampir_featuresTest$Label <- as.factor(ampir_test_df$Label)
rownames(ampir_featuresTest) <- NULL


#ampir_pred <- predict(ampir_svm_model, ampir_featuresTest)

ampir_AMPs <- predict_amps(ampir_test_df, min_len = 4, model = ampir_svm_model)


ampir_AMPs %>%
  select(one_of("seq_name","Label","prob_AMP")) %>% 
  rename("ID"="seq_name","target"="Label","probability"="prob_AMP") %>% 
  mutate(prediction=NA) %>% 
  write.csv(file=output_file,row.names = FALSE,quote = FALSE)
