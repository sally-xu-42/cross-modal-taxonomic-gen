library(tidyverse)

question_scores <- read_csv("kanishka_res/llm-backbone-yn-filtering/allenai_Olmo-3-1025-7B.csv")

stimuli <- stream_in(file("llm-backbone-exp-data/yn_questions.jsonl")) %>%
  as_tibble() %>%
  mutate(
    idx = row_number()-1
  ) %>% inner_join(question_scores)

stimuli %>%
  distinct(hyponym, hypernym)

raw_results_vlm <- fs::dir_ls("kanishka_res/", regexp = "*.csv") %>%
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

llm_stimuli <- stream_in(file("llm-backbone-exp-data/yn_questions.jsonl")) %>%
  as_tibble() %>%
  mutate(
    idx = row_number()-1
  ) %>% inner_join(question_scores)

raw_results_llm <- fs::dir_ls("kanishka_res/llm-backbone-yn/", regexp="*.csv") %>%
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
  )

llm_corrects <- raw_results_llm %>%
  select(model, hyponym, hypernym, phrasing_id, correct) %>%
  distinct() %>%
  pivot_wider(names_from = phrasing_id, values_from = correct, names_prefix = "phrasing_") %>%
  mutate(
    llm_max_correct = phrasing_1 | phrasing_2 | phrasing_3 | phrasing_4,
    llm_joint_correct = phrasing_1 & phrasing_2 & phrasing_3 & phrasing_4
  )

llm_corrects_filtered <- raw_results_llm %>%
  group_by(model, item, q_type) %>%
  filter(score == max(score)) %>%
  ungroup() %>%
  select(model, hyponym, hypernym, phrasing_id, correct) %>%
  distinct()

llm_corrects_filtered %>% count(model, phrasing_id)

llm_corrects_filtered %>%
  group_by(model) %>%
  summarize(
    n = n(),
    accuracy = mean(correct)
  )
  # select(model, hyponym, hypernym, phrasing_id, correct) %>%
  # distinct() %>%
  # pivot_wider(names_from = phrasing_id, values_from = correct, names_prefix = "phrasing_") %>%
  # mutate(
  #   llm_max_correct = phrasing_1 | phrasing_2 | phrasing_3 | phrasing_4,
  #   llm_joint_correct = phrasing_1 & phrasing_2 & phrasing_3 & phrasing_4
  # )

llm_corrects %>% 
  group_by(model) %>% 
  summarize(max_acc = mean(llm_max_correct), joint = mean(llm_joint_correct))

llm_corrects %>% 
  group_by(model, hypernym) %>% 
  summarize(max_acc = mean(llm_max_correct), joint = mean(llm_joint_correct)) %>%
  mutate(
    diff = max_acc - joint
  ) %>% View("llm")


vlm_corrects <- raw_results_vlm %>%
  group_by(model, vision_encoder, concept, hypernym) %>%
  summarise(
    n = n(),
    vlm_joint_correct = mean(answer == predicted_answer) == 1, # model is correct for all images
    vlm_max_correct = mean(answer == predicted_answer) > 0 # model is correct at least for one image
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

vlm_corrects %>% 
  group_by(model, vision_encoder, hypernym) %>% 
  summarize(max_acc = mean(vlm_max_correct), joint = mean(vlm_joint_correct)) %>%
  mutate(
    diff = max_acc - joint
  ) %>% View("vlm")

llm_corrects %>%
  inner_join(vlm_corrects) %>%
  group_by(model, vision_encoder) %>%
  summarize(
    max_agreement = mean(llm_max_correct == vlm_max_correct), # model was correct for all surface forms
    joint_agreement = mean(llm_joint_correct == vlm_joint_correct) # model was correct for at least one image
  ) %>%
  ungroup() %>%
  mutate(
    hypernym = "Overall"
  ) %>%
  ggplot(aes(vision_encoder, joint_agreement, color = model)) +
  geom_point(size = 2) +
  # facet_wrap(~vision_encoder, nrow = 2) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format()) +
  theme_bw(base_size = 16) +
  theme(
    # axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    # x = "Hypernym",
    y = "Agreement between\nVLM and LM predictions",
    x = "Vision Encoder",
    color = "Model"
  )

llm_corrects %>%
  inner_join(vlm_corrects) %>%
  group_by(model, vision_encoder, hypernym) %>%
  summarize(
    n = n(),
    max_agreement = mean(llm_max_correct == vlm_max_correct), # model was correct for all surface forms
    joint_agreement = mean(llm_joint_correct == vlm_joint_correct) # model was correct for at least one image
  ) %>%
  ungroup() %>%
  filter(vision_encoder == "DINOv2") %>% 
  pivot_longer(max_agreement:joint_agreement, names_to = "agreement_type", values_to = "agreement") %>%
  mutate(
    agreement_type = case_when(
      agreement_type == "max_agreement" ~ "Maximum",
      agreement_type == "joint_agreement" ~ "Joint"
    ),
    hypernym = factor(hypernym),
    hypernym = fct_reorder(hypernym, n)
  ) %>% 
  # ggplot(aes(hypernym, agreement, color = model, fill = model, shape = model)) +
  ggplot(aes(hypernym, agreement, color = agreement_type, fill = agreement_type, shape = agreement_type)) +
  geom_point(size = 2) +
  # facet_wrap(~agreement_type, nrow = 2) +
  facet_wrap(~model, nrow = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(suffix = "")) +
  scale_shape_manual(values = c(21,23)) +
  scale_color_manual(values = c("#e7298a", "#66a61e"), aesthetics=c("fill", "color")) +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    x = "Hypernym Category",
    y = "Agreement between\nVLM and LM predictions (%)",
    shape = "Agreement Type",
    color = "Agreement Type",
    fill = "Agreement Type"
  )

ggsave("plots/lm-agreement-seed42.pdf", height = 7.62, width = 14.88, dpi=300, device=cairo_pdf)


llm_corrects_filtered %>%
  inner_join(vlm_corrects) %>%
  group_by(model, vision_encoder, hypernym) %>%
  summarize(
    n = n(),
    max_agreement = mean(correct == vlm_max_correct), # model was correct for all surface forms
    joint_agreement = mean(correct == vlm_joint_correct) # model was correct for at least one image
  ) %>%
  ungroup() %>%
  filter(vision_encoder == "DINOv2") %>% 
  pivot_longer(max_agreement:joint_agreement, names_to = "agreement_type", values_to = "agreement") %>%
  mutate(
    agreement_type = case_when(
      agreement_type == "max_agreement" ~ "Maximum",
      agreement_type == "joint_agreement" ~ "Joint"
    ),
    hypernym = factor(hypernym),
    hypernym = fct_reorder(hypernym, n)
  ) %>% 
  # ggplot(aes(hypernym, agreement, color = model, fill = model, shape = model)) +
  ggplot(aes(hypernym, agreement, color = agreement_type, fill = agreement_type, shape = agreement_type)) +
  geom_point(size = 2) +
  # facet_wrap(~agreement_type, nrow = 2) +
  facet_wrap(~model, nrow = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(suffix = "")) +
  scale_shape_manual(values = c(21,23)) +
  scale_color_manual(values = c("#e7298a", "#66a61e"), aesthetics=c("fill", "color")) +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    x = "Hypernym Category",
    y = "Agreement between\nVLM and LM predictions (%)",
    shape = "Agreement Type",
    color = "Agreement Type",
    fill = "Agreement Type"
  )

ggsave("plots/lm-agreement-seed42.pdf", height = 7.62, width = 14.88, dpi=300, device=cairo_pdf)



llm_corrects %>%
  inner_join(vlm_corrects) %>%
  group_by(model, vision_encoder) %>%
  summarize(
    max_correct_correct = mean(vlm_max_correct == TRUE & llm_max_correct == TRUE),
    max_correct_incorrect = mean(vlm_max_correct == TRUE & llm_max_correct == FALSE),
    max_incorrect_correct = mean(vlm_max_correct == FALSE & llm_max_correct == TRUE),
    max_incorrect_incorrect = mean(vlm_max_correct == FALSE & llm_max_correct == FALSE),
    joint_correct_correct = mean(vlm_joint_correct == TRUE & llm_joint_correct == TRUE),
    joint_correct_incorrect = mean(vlm_joint_correct == TRUE & llm_joint_correct == FALSE),
    joint_incorrect_correct = mean(vlm_joint_correct == FALSE & llm_joint_correct == TRUE),
    joint_incorrect_incorrect = mean(vlm_joint_correct == FALSE & llm_joint_correct == FALSE),
  ) %>% View()

overall_agreement <- llm_corrects %>%
  inner_join(vlm_corrects) %>%
  group_by(model, vision_encoder, hypernym) %>%
  summarize(
    n = n(),
    max_agreement = mean(llm_max_correct == vlm_max_correct), # model was correct for all surface forms
    joint_agreement = mean(llm_joint_correct == vlm_joint_correct) # model was correct for at least one image
  ) %>%
  ungroup() %>%
  filter(vision_encoder == "DINOv2") %>% 
  pivot_longer(max_agreement:joint_agreement, names_to = "measure", values_to = "value") %>%
  mutate(
    measure = case_when(
      measure == "max_agreement" ~ "Maximum",
      measure == "joint_agreement" ~ "Joint"
    )
  ) %>%
  group_by(model, measure) %>%
  summarize(
    n = n(),
    sd = sd(value),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    value = mean(value)
  )

llm_corrects_filtered %>%
  inner_join(vlm_corrects) %>%
  group_by(model, vision_encoder, hypernym) %>%
  summarize(
    n = n(),
    max_agreement = mean(correct == vlm_max_correct), # model was correct for all surface forms
    joint_agreement = mean(correct == vlm_joint_correct) # model was correct for at least one image
  ) %>%
  ungroup() %>%
  filter(vision_encoder == "DINOv2") %>% 
  pivot_longer(max_agreement:joint_agreement, names_to = "measure", values_to = "value") %>%
  mutate(
    measure = case_when(
      measure == "max_agreement" ~ "Maximum",
      measure == "joint_agreement" ~ "Joint"
    )
  ) %>%
  group_by(model, measure) %>%
  summarize(
    n = n(),
    sd = sd(value),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    value = mean(value)
  )

overall_llm_accuracy <- llm_corrects %>% 
  group_by(model, hypernym) %>% 
  summarize(max_acc = mean(llm_max_correct), joint = mean(llm_joint_correct)) %>%
  ungroup() %>%
  pivot_longer(max_acc:joint, names_to = "measure", values_to = "value") %>%
  mutate(
    measure = case_when(
      measure == "max_acc" ~ "Maximum",
      measure == "joint" ~ "Joint"
    )
  ) %>%
  group_by(model, measure) %>%
  summarize(
    n = n(),
    sd = sd(value),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    value = mean(value)
  )

bind_rows(
  overall_agreement %>% mutate(exp = "Agreement"),
  overall_llm_accuracy %>% mutate(exp = "LM Accuracy")
) %>%
  ggplot(aes(model, value, color = measure, fill = measure, shape = measure)) +
  geom_point(size = 2, position=position_dodge(0.5)) +
  geom_linerange(aes(ymin=value-conf,ymax=value+conf), position=position_dodge(0.5)) +
  facet_wrap(~exp) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_shape_manual(values = c(21,23)) +
  scale_color_manual(values = c("#e7298a", "#66a61e"), aesthetics=c("fill", "color")) +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    # axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) 
