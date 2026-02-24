library(tidyverse)
library(jsonlite)

stimuli <- stream_in(file("llm-backbone-exp-data/yn_questions.jsonl")) %>%
  as_tibble() %>%
  mutate(
    idx = row_number()-1
  )

category_results_llm <- fs::dir_ls("kanishka_res/llm-backbone-yn/", regexp="*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_remove(model, "kanishka_res/llm-backbone-yn/"),
    model = str_remove(model, "(Qwen_|Llama_)"),
    model = str_remove(model, ".csv")
  ) %>%
  inner_join(stimuli) %>%
  select(-idx) %>%
  mutate(
    prediction = case_when(
      rank_yes < rank_no ~ "Yes",
      TRUE ~ "No"
    ),
    correct = prediction == label
  ) %>%
  select(model, hyponym, hypernym, phrasing_id, correct) %>%
  distinct() %>%
  pivot_wider(names_from = phrasing_id, values_from = correct, names_prefix = "phrasing_") %>%
  group_by(model, hypernym) %>%
  summarise(
    max_acc = mean(phrasing_1 | phrasing_2 | phrasing_3 | phrasing_4),
    joint_acc = mean(phrasing_1 & phrasing_2 & phrasing_3 & phrasing_4)
  ) %>%
  ungroup()

llm_corrects <- fs::dir_ls("kanishka_res/llm-backbone-yn/", regexp="*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_remove(model, "kanishka_res/llm-backbone-yn/"),
    model = str_remove(model, "(Qwen_|Llama_)"),
    model = str_remove(model, ".csv")
  ) %>%
  inner_join(stimuli) %>%
  select(-idx) %>%
  mutate(
    prediction = case_when(
      rank_yes < rank_no ~ "Yes",
      TRUE ~ "No"
    ),
    correct = prediction == label
  ) %>%
  select(model, hyponym, hypernym, phrasing_id, correct) %>%
  distinct() %>%
  pivot_wider(names_from = phrasing_id, values_from = correct, names_prefix = "phrasing_") %>%
  mutate(
    llm_max_correct = phrasing_1 | phrasing_2 | phrasing_3 | phrasing_4,
    llm_joint_correct = phrasing_1 & phrasing_2 & phrasing_3 & phrasing_4
  )

llm_corrects %>%
  inner_join(vlm_corrects) %>%
  group_by(model, vision_encoder, hypernym) %>%
  summarize(
    max_agreement = mean(llm_max_correct == vlm_max_correct),
    joint_agreement = mean(llm_joint_correct == vlm_joint_correct)
  ) %>%
  # filter(model == "Qwen3-0.6B", vision_encoder == "DINOv2") %>%
  # filter(model == "1b", n > 10) %>%
  # filter(vision_encoder == "DINOv2") %>%
  # filter(model == "Qwen3-0.6B") %>%
  # filter(model == "1b", vision_encoder == "DINOv2", n > 10) %>%
  mutate(
    # hypernym = factor(hypernym),
    # hypernym = fct_reorder(hypernym, joint_agreement, .desc = TRUE)
  ) %>%
  # ggplot(aes(hypernym, mean, color = vision_encoder)) +
  ggplot(aes(hypernym, joint_agreement, color = model, shape = vision_encoder)) +
  geom_point(size = 2) +
  facet_wrap(~vision_encoder, nrow = 2) +
  # geom_linerange(aes(ymin = mean-cb, ymax=mean+cb)) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format()) +
  theme_bw(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    legend.position = "top"
  ) +
  labs(
    x = "Hypernym",
    y = "Agreement between\nVLM and LM predictions",
    shape = "Vision Encoder",
    color = "Model"
  )

category_results_vlm %>%
  inner_join(category_results_llm) %>%
  group_by(model, vision_encoder) %>%
  nest() %>%
  mutate(
    cor = map_df(data, function(x){
      cor.test(x$joint_acc_vlm, x$max_acc, method = "spearman") %>%
        broom::tidy()
    })
  )

category_results_llm %>%
  distinct(hypernym) %>%
  anti_join(category_results_vlm %>%
              distinct(hypernym))
