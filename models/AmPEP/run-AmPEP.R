#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(dplyr)
library(randomForest)
library(tidysq)

groupings <- list(
  hydrophobicity = list(C1 = c("R", "K", "E", "D", "Q", "N"),
                        C2 = c("G", "A", "S", "T", "P", "H", "Y"),
                        C3 = c("C", "L", "V", "I", "M", "F", "W")),
  van_der_Waals_volume = list(C1 = c("G", "A", "S", "T", "P", "D"),
                              C2 = c("N", "V", "E", "Q", "I", "L"),
                              C3 = c("M", "H", "K", "F", "R", "Y", "W")),
  polarity = list(C1 = c("L", "I", "F", "W", "C", "M", "V", "Y"),
                  C2 = c("P", "A", "T", "G", "S"),
                  C3 = c("H", "Q", "R", "K", "N", "E", "D")),
  polarizability = list(C1 = c("G", "A", "S", "D", "T"),
                        C2 = c("C", "P", "N", "V", "E", "Q", "I", "L"),
                        C3 = c("K", "M", "H", "F", "R", "Y", "W")),
  charge = list(C1 = c("K", "R"),
                C2 = c("A", "N", "C", "Q", "G", "H", "I", "L", "M", "F", "P", "S", "T", "W", "Y", "V"),
                C3 = c("D", "E")),
  secondary_struct = list(C1 = c("E", "A", "L", "M", "Q", "K", "R", "H"),
                          C2 = c("V", "I", "Y", "C", "W", "F", "T"),
                          C3 = c("G", "N", "P", "S", "D")),
  solvent_accessibility = list(C1 = c("A", "L", "F", "C", "G", "I", "V", "W"),
                               C2 = c("P", "K", "Q", "E", "N", "D"),
                               C3 = c("M", "P", "S", "T", "H", "Y"))
)

prepare_features <- function(input_file) {
  sequences_df <- read_fasta(input_file, alphabet = "ami_ext") %>%
    mutate(sq = remove_ambiguous(sq)) %>%
    filter(get_sq_lengths(sq) > 0)
  
  # Calculating Distribution descriptors
  ret <- lapply(groupings, function(property) {
    ret <- lapply(property, function(group) {
      ret <- sequences_df %>% pull(sq) %>% sqapply(function(seq) {
        found <- seq %in% group
        found_count <- sum(found)
        c(first = which.min(found),
          percent25 = floor(found_count * 0.25),
          percent50 = floor(found_count * 0.5),
          percent75 = floor(found_count * 0.75),
          percent100 = found_count)/length(seq)
      })
      do.call(rbind, ret)
    })
    ret <- do.call(cbind, ret)
    colnames(ret) <- paste0(rep(c("C1_", "C2_", "C3_"), each = 5), colnames(ret))
    ret
  })
  ret <- do.call(cbind, ret)
  colnames(ret) <- paste0(rep(names(groupings), each = 5*3), "_", colnames(ret))
  
  bind_cols(ret, ID = sequences_df[["name"]]) %>%
    select(ID,
           charge_C2_first, charge_C3_first, charge_C3_percent25,
           charge_C2_percent50, charge_C3_percent50, charge_C1_percent75,
           charge_C3_percent75, charge_C2_percent100, charge_C3_percent100,
           hydrophobicity_C3_first, hydrophobicity_C1_percent100,
           van_der_Waals_volume_C3_first,
           polarity_C1_first, polarity_C2_percent25, polarity_C2_percent50,
           polarity_C2_percent75,
           polarizability_C3_first,
           secondary_struct_C1_first, secondary_struct_C1_percent100,
           solvent_accessibility_C1_first, solvent_accessibility_C3_first,
           solvent_accessibility_C1_percent50, solvent_accessibility_C2_percent100
    ) %>%
    mutate(target = as.factor(ifelse(grepl("AMP=1", ID), 1, 0)))
}

train_df <- prepare_features(args[1])
test_df <- prepare_features(args[2])

trained_rf <- randomForest(target ~ ., select(train_df, -ID),
                           ntree = 100)
preds <- predict(trained_rf, select(test_df, -c(ID, target)),
                 type = "prob") %>%
  as_tibble() %>%
  transmute(prediction = ifelse(`1` > 0.5, "1", "0"),
            probability = pmax(`1`, `0`))

res <- test_df %>%
  select(ID, target) %>%
  bind_cols(preds)

write.csv(res, args[3], row.names = FALSE)
