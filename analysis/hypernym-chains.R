library(tidyverse)

pairs <- read_csv("../things-util/data/category-pairs.csv") %>%
  mutate(score = 1)

pairs %>%
  distinct(category) %>%
  write_csv("data/unique-categories.csv")

pairs %>%
  # filter(category %in% c("clothing", "outerwear", "headwear", "jewelry", "clothing accessory")) %>%
  # filter(category %in% c("animal", "farm animal")) %>%
  # filter(category %in% c("vegetable", "plant")) %>%
  filter(category %in% c("farm animal", "insect")) %>%
  pivot_wider(names_from = category, values_from = score, values_fill = 0) %>%
  View()


pairs %>%
  pivot_wider(names_from = category, values_from = score, values_fill = 0) %>%
  pivot_longer(animal:`personal hygiene item`, names_to = "category", values_to = "is_hypernym") %>%
  widyr::pairwise_cor(category, unique_id, is_hypernym) %>%
  View("pairwise")
