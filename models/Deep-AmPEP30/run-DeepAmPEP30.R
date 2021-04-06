#!/usr/bin/env Rscript

# input arguments loading ----

args <- commandArgs(trailingOnly=TRUE)
train_path <- args[1]
test_path <- args[2]
output_path <- args[3]


# packages loading ----

library("keras")
library("tidysq")
library("dplyr")
library("stringi")
library("ftrCOOL") # to compute PseKRAAC


# feature engineering ----

tabularize_sequences <- function(input_path) {
  # for one sequence ----
  # a putative value (the paper does not mention the value of this param explicitly)
  # the value 1 is deduced since it results in 86 features
  # (the same number of features is mentioned in the paper)
  # the larger values of k gives more features
  k <- 1
  feature_eng_params <- list(list(f=ftrCOOL::PseKRAAC_T3A, clusters=19),
                             list(f=ftrCOOL::PseKRAAC_T7, clusters=15),
                             list(f=ftrCOOL::PseKRAAC_T8, clusters=17),
                             list(f=ftrCOOL::PseKRAAC_T11, clusters=17),
                             list(f=ftrCOOL::PseKRAAC_T11, clusters=18))
  
  prepare_features_for_one_seq <- function(seq) {
    lapply(feature_eng_params, function(fe_entry) {
      fe_entry$f(seq, Grp=fe_entry$clusters, k=k)
    }) %>% do.call(cbind, .)
  }
  
  # all sequences processing ----
  sequences_df <- read_fasta(input_path, alphabet="ami_bsc")
  sequences_df %>%
    pull(sq) %>%
    sqapply(prepare_features_for_one_seq,
            single_string = TRUE) %>%
    do.call(rbind, .) %>%
    as_tibble() %>%
    bind_cols(sequences_df, .) %>%
    mutate(label = if_else(stringi::stri_match_last_regex(name, "(?<=AMP\\=)0|1") == "1", 1, 0)) %>% 
    select(-sq) %>% 
    as.data.frame
}

prepare_model_compliant_data <- function(df) {
  list(
    x=df %>%
      select(-name, -label) %>%
      as.matrix %>%
      array(dim=c(dim(.), 1)),
    y=df$label
  )
}


# model training ----

prepare_model <- function(features_num, seed=1234) {
  model <- keras_model_sequential()
  model %>% 
    layer_batch_normalization(input_shape = c(features_num, 1)) %>% 
    layer_conv_1d(filters = 128,
                  kernel_size = 3,
                  strides = 1,
                  activation = "relu",
                  padding = "same") %>%
    layer_max_pooling_1d(pool_size = 2,
                         strides = 2) %>%
    layer_dropout(rate = 0.2,
                  seed = seed) %>%
    layer_conv_1d(filters = 128,
                  kernel_size = 3,
                  strides = 1,
                  activation = "relu",
                  padding = "same") %>%
    layer_max_pooling_1d(pool_size = 2,
                         strides = 2) %>%
    layer_dropout(rate = 0.2,
                  seed = seed) %>%
    layer_flatten() %>%
    layer_dense(units = 10,
                activation = "relu") %>%
    layer_dense(units = 1,
                activation = "sigmoid")
  
  model %>% compile(loss = "binary_crossentropy",
                    optimizer = optimizer_rmsprop(),
                    metrics=c("accuracy"))
  model
}

train_model <- function(train_data, seed=1234) {
  model <- prepare_model(features_num=ncol(train_data$x),
                         seed=seed)
  
  model %>% fit(train_data$x,
                train_data$y,
                batch_size = 64,
                epochs = 100)  # the paper does not specify the exact value of validation_split
  
  model
}

train_data <- tabularize_sequences(train_path) %>% 
  prepare_model_compliant_data

trained_model <- train_model(train_data)

# model testing ----

test_sequences <- tabularize_sequences(test_path)
test_data <- prepare_model_compliant_data(test_sequences)
res <- data.frame(
  ID=test_sequences$name,
  target=test_data$y,
  probability=trained_model %>% predict_proba(test_data$x),
  prediction=trained_model %>% predict_classes(test_data$x))

write.csv(res, args[3], row.names = FALSE)
