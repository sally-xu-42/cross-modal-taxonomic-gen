library(tidyverse)

raw_results <- fs::dir_ls("kanishka_res/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_remove(file, "kanishka_res/align-"),
    model = str_remove(model, "-\\d{1,3}_unseen.csv"),
    vision_encoder = case_when(
      str_detect(model, "dinosiglip") ~ "DINO+SigLIP",
      TRUE ~ "DINOv2"
    ),
    model = str_remove(model, "(dinosiglip|dinov2)\\+"),
    model = str_remove(model, "-things"),
    answer = str_to_lower(answer),
    predicted_answer = str_to_lower(predicted_answer),
    hypernym = case_when(
      hypernym == "arts and crafts item" ~ "arts and crafts supply",
      hypernym == "car part" ~ "part of car",
      hypernym == "hardware item" ~ "hardware",
      hypernym == "home decor item" ~ "home decor",
      hypernym == "item of clothing" ~ "clothing",
      hypernym == "musical" ~ "musical instrument",
      hypernym == "office supply item" ~ "office supply",
      hypernym == "piece of jewelry" ~ "jewelry",
      hypernym == "piece of women's clothing" ~ "women's clothing",
      hypernym == "protective clothing item" ~ "protective clothing",
      hypernym == "source of light" ~ "lighting",
      TRUE ~ hypernym
    )
  )


raw_results %>%
  group_by(model, vision_encoder) %>%
  summarize(
    accuracy = mean(answer == predicted_answer)
  ) 
  

vlm_corrects <- raw_results %>%
  group_by(model, vision_encoder, concept, hypernym) %>%
  summarise(
    n = n(),
    vlm_joint_correct = mean(answer == predicted_answer) == 1,
    vlm_max_correct = mean(answer == predicted_answer) > 0
  ) %>%
  ungroup() %>%
  filter(model %in% c("500m", "1b")) %>%
  mutate(
    model = case_when(
      model == "1b" ~ "Qwen3-1.7B",
      model == "500m" ~ "Qwen3-0.6B"
    )
  ) %>%
  rename(hyponym = concept)


category_results_vlm <- raw_results %>%
  group_by(model, vision_encoder, concept, hypernym) %>%
  summarise(
    n = n(),
    correct = mean(answer == predicted_answer)
  ) %>%
  ungroup() %>%
  group_by(model, vision_encoder, hypernym) %>%
  summarize(
    n = n(),
    joint_acc_vlm = mean(correct)
  ) %>%
  ungroup() %>%
  filter(model %in% c("500m", "1b")) %>%
  mutate(
    model = case_when(
      model == "1b" ~ "Qwen3-1.7B",
      model == "500m" ~ "Qwen3-0.6B"
    )
  )

raw_results %>%
  group_by(model, vision_encoder, concept, hypernym) %>%
  summarise(
    accuracy = mean(answer == predicted_answer)
  ) %>%
  ungroup() %>%
  group_by(model, vision_encoder, hypernym) %>%
  summarize(
    n = n(),
    sd = sd(accuracy),
    cb = qt(0.05/2, n-1, lower.tail = FALSE) * sd/sqrt(n),
    mean = mean(accuracy)
  ) %>%
  ungroup() %>%
  filter(model == "500m", vision_encoder == "DINOv2") %>%
  # filter(model == "1b", n > 10) %>%
  # filter(model == "1b", vision_encoder == "DINOv2", n > 10) %>%
  mutate(
    hypernym = factor(hypernym),
    hypernym = fct_reorder(hypernym, mean, .desc = TRUE)
  ) %>%
  # ggplot(aes(hypernym, mean, color = vision_encoder)) +
  ggplot(aes(hypernym, mean)) +
  geom_point(size = 2) +
  geom_linerange(aes(ymin = mean-cb, ymax=mean+cb)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
  ) +
  labs(
    x = "Hypernym",
    y = "Generalization Accuracy"
  )

