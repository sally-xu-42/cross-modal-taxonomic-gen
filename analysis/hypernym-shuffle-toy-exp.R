library(tidyverse)

# original <-  %>%
#   pivot_longer(bird:weapon, names_to = "category", values_to = "f1")


held_out <- bind_rows(
  read_csv("results/main-results/macro-f1s/cat_sampled_unseen_f1_scores.csv") %>%
    mutate(exp = "original"),
  read_csv("results/main-results/macro-f1s/cat_unseen_f1_scores.csv") %>%
    mutate(exp = "shuffled"),
  read_csv("results/main-results/macro-f1s/catwise_hyp_guess_f1_scores.csv") %>%
    mutate(exp= "random")
) %>%
  pivot_longer(bird:weapon, names_to = "category", values_to = "f1")
  
held_out
  group_by(exp) %>%
  summarize(
    n = n(),
    sd = sd(f1),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    f1 = mean(f1)
  )

# t.test(held_out)
original <- held_out %>%
  filter(exp=="original") %>%
  pull(f1)

shuffled <- held_out %>%
  filter(exp=="shuffled") %>%
  pull(f1)

t.test(original, shuffled)

bind_rows(
  read_csv("results/main-results/macro-f1s/cat_sampled_leaf_f1_scores.csv") %>%
    mutate(exp = "original"),
  read_csv("results/main-results/macro-f1s/cat_leaf_f1_scores.csv") %>%
    mutate(exp = "shuffled"),
  read_csv("results/main-results/macro-f1s/catwise_leaf_guess_f1_scores.csv") %>%
    mutate(exp= "random")
) %>%
  pivot_longer(`air mattress`:yacht, names_to = "category", values_to = "f1") %>%
  group_by(exp) %>%
  summarize(
    f1 = mean(f1)
  )
