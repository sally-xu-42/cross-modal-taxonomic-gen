library(tidyverse)
library(jsonlite)

question_scores <- read_csv("results/llm-backbone-yn-filtering/allenai_Olmo-3-1025-7B.csv")
category_lexicon <- read_csv("data/llm-backbone-exp-data/category-lexicon.csv")
visual_coherence <- read_csv("results/main-results/visual_coherence_results.csv") %>%
  inner_join(category_lexicon %>%
              select(category = surface_form, hypernym = unique_id) ) %>%
  select(-n_images)

# category_lexicon %>%
#   select(category = surface_form, hypernym = unique_id) %>%
#   anti_join(visual_coherence %>% distinct(category))

# %>%
  # inner_join(category_lexicon %>% select(category = surface_form, hypernym = unique_id))

stimuli <- stream_in(file("data/llm-backbone-exp-data/all_questions.jsonl")) %>%
  as_tibble() %>%
  mutate(
    idx = row_number()-1
  ) %>%
  inner_join(question_scores)

final_stimuli <- stimuli %>%
  group_by(item, q_type, type_id) %>%
  filter(score == max(score)) %>%
  select(-score) %>%
  ungroup()

item_hyps <- final_stimuli %>%
  filter(q_type == "positive") %>%
  distinct(item, hyponym, pos_hypernym = hypernym)

raw_results_llm <- fs::dir_ls("results/llm-backbone-yn/", regexp="*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_remove(model, "results/llm-backbone-yn/"),
    model = str_remove(model, "(Qwen_|Llama_)"),
    model = str_remove(model, ".csv")
  ) %>%
  inner_join(final_stimuli) %>%
  select(-idx) %>%
  mutate(
    prediction = case_when(
      rank_yes < rank_no ~ "Yes",
      TRUE ~ "No"
    ),
    correct = prediction == answer
  )

# category-wise acc 
coherence_backbone <- raw_results_llm %>%
  inner_join(item_hyps) %>%
  group_by(model, pos_hypernym) %>%
  summarize(
    backbone_acc = mean(correct)
  ) %>%
  ungroup() %>%
  rename(hypernym = pos_hypernym) %>%
  inner_join(visual_coherence)

coherence_backbone %>%
  write_csv("results/coherence_backbone.csv")

# model accs
raw_results_llm %>%
  mutate(
    answer = factor(answer),
    prediction = factor(prediction)
  ) %>%
  group_by(model) %>%
  summarize(
    acc = mean(correct),
    f1 = yardstick::f_meas_vec(answer, prediction, estimator = "macro")
  )

answers <- raw_results_llm %>%
  filter(model == "Qwen3-0.6B") %>%
  pull(answer) %>%
  factor()
  # factor(levels = c("Yes", "No"))

predictions <- raw_results_llm %>%
  filter(model == "Qwen3-0.6B") %>%
  pull(prediction) %>%
  factor()
  # factor(levels = c("Yes", "No"))

# majority <- factor(rep(c("No"), length(answers)), levels = c("Yes", "No"))
majority <- factor(rep(c("No"), length(answers)), levels = c("No", "Yes"))

answers

yardstick::f_meas_vec(answers, predictions, estimator = "macro")
yardstick::f_meas_vec(answers, majority, estimator = "micro")





