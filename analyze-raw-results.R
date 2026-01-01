library(tidyverse)
library(lmerTest)

read_results <- function(path) {
  read_csv(path) %>%
    mutate(
      seed = case_when(
        str_detect(run_name, "-218($|-random)") ~ 218,
        str_detect(run_name, "-42($|-random)") ~ 42,
        str_detect(run_name, "-7($|-random)") ~ 7
      ),
      random = str_detect(run_name, "-random$"),
      vision_encoder = case_when(
        str_detect(run_name, "dinosiglip") ~ "DINO+SigLIP",
        str_detect(run_name, "dinov2") ~ "DINOv2",
        TRUE ~ "SigLIP"
      ),
      lm = case_when(
        str_detect(run_name, "1b-llama-chat") ~ "Llama2-Instruct",
        str_detect(run_name, "1b-llama-things") ~ "Llama2",
        str_detect(run_name, "500m") ~ "Qwen3-0.6B",
        TRUE ~ "Qwen3-1.7B"
      ),
      shuffled = case_when(
        str_detect(run_name, "local-shuffled") ~ "Within-category",
        !str_detect(run_name, "shuffled") ~ "Original",
        TRUE ~ "Across-category"
      ),
      ablation_type = case_when(
        str_detect(run_name, "-abl\\d{2}cat-") ~ "Systematic",
        str_detect(run_name, "-abl\\d{2}-") ~ "Random",
        TRUE ~ "Full"
      ),
      ablation_amt = case_when(
        str_detect(run_name, "-abl\\d{2}cat-") ~ as.numeric(str_extract(run_name, "(?<=abl)(.*)(?=cat)")),
        str_detect(run_name, "-abl\\d{2}-") ~ str_extract(run_name, "(?<=abl)(.*)(?=-(\\d{1,2}|local|shuffled))") %>%
          str_extract("\\d{2}") %>%
          as.numeric(),
        TRUE ~ 100
      ),
      seen_hypernyms = case_when(
        str_detect(run_name, "-abl\\d{2}cat-") ~ 100 * ((53-ablation_amt)/53),
        str_detect(run_name, "-abl\\d{2}-") ~ 100-ablation_amt,
        TRUE ~ 100-ablation_amt
      )
    )
}

# leaf <- read_csv("kanishka_res/raw-results-google-drive/all_leaf_evals.csv")
# leaf_shuffled <- read_csv("kanishka_res/raw-results-google-drive/shuffled_leaf_evals.csv")

leaf <- read_results("kanishka_res/raw-results-google-drive/all_leaf_evals.csv")
unseen <- read_results("kanishka_res/raw-results-google-drive/all_unseen_evals.csv")
seen <- read_results("kanishka_res/raw-results-google-drive/all_seen_evals.csv")
unseen_shuffled <- read_results("kanishka_res/raw-results-google-drive/shuffled_unseen_evals.csv")

unseen_shuffled %>% filter(vision_encoder=="DINOv2", str_detect(lm, "Qwen")) %>%
  filter(is.na(ablation_amt))

leaf %>% count(seed, random, vision_encoder, lm, shuffled, ablation_type, ablation_amt)

leaf %>%
  select(-overall, -total_illegal_ratio) %>%
  pivot_longer(aardvark:zucchini, names_to = "leaf", values_to = "acc")


# 100% ablation accuracy comparison between DINO and SigLIP
dino_vs_siglip <- unseen %>%
  select(-overall, -total_illegal_ratio) %>%
  pivot_longer(mammal:condiment, names_to = "hypernym", values_to = "acc") %>%
  filter(vision_encoder %in% c("SigLIP", "DINOv2"), ablation_type == "Full", random == F) %>%
  filter(str_detect(lm, "Qwen"))

fit_dinosiglip <- lmer(
  acc ~ vision_encoder * lm + 
    (1 + vision_encoder*lm | seed) + 
    (1 + vision_encoder*lm | hypernym),
  data = dino_vs_siglip %>% 
    mutate(
      lm = case_when(
        lm == "Qwen3-0.6B" ~ 1,
        TRUE ~ -1
      ),
      vision_encoder = case_when(
        vision_encoder == "DINOv2" ~ 1,
        TRUE ~ -1
      )
    )
)

summary(fit_dinosiglip)

# no significant effect of vision encoder diff = 0.01, t = 0.839, p = 0.487

dino_vs_siglip %>%
  group_by(lm, vision_encoder, hypernym) %>%
  summarize(acc = mean(acc)) %>%
  ungroup() %>%
  group_by(lm, vision_encoder) %>%
  summarize(
    n = n(),
    sd = sd(acc),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    acc = mean(acc)
  ) %>%
  ggplot(aes(lm, acc, color = vision_encoder, fill = vision_encoder, shape = vision_encoder)) +
  geom_point(size = 2, position = position_dodge(0.5))+
  geom_linerange(aes(ymin = acc-conf, ymax = acc+conf), position = position_dodge(0.5)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.5) +
  scale_y_continuous(limits = c(0.5, 1), labels = scales::percent_format(suffix = "")) +
  scale_shape_manual(values = c(21,23)) +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    # legend.margin = 
    # legend.justification='left',
    # legend.direction='horizontal'
    legend.box.margin = margin(0,0,0,-30),
    axis.title = element_text(size = 16),
    legend.title = element_text(size=16),
  ) +
  labs(
    x = "LM Backbone",
    y = "Accuracy on\nunseen images (%)",
    color = "Image Encoder",
    fill = "Image Encoder",
    shape = "Image Encoder"
  )

# 446W x 360H

ggsave("plots/dino-vs-siglip.pdf", width = 4.45, height = 3.6, dpi=300, device=cairo_pdf)

# Dino across ablations
exp1_ablation_results_raw <- bind_rows(
  unseen %>%
    select(-overall, -total_illegal_ratio) %>%
    pivot_longer(mammal:condiment, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Held out Hypernyms"),
  seen %>%
    select(-overall, -total_illegal_ratio) %>%
    pivot_longer(musical:headwear, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Seen Hypernyms"),
  leaf %>%
    select(-overall, -total_illegal_ratio) %>%
    pivot_longer(aardvark:zucchini, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Leaves")
) %>%
  mutate(
    experiment = factor(experiment, levels = c("Leaves", "Seen Hypernyms", "Held out Hypernyms"))
  ) %>%
  filter(vision_encoder %in% c("DINOv2")) %>%
  filter(str_detect(lm, "Qwen")) %>% 
  filter(!is.na(acc)) %>%
  group_by(experiment, lm, seen_hypernyms, ablation_type, category, random) %>%
  summarize(acc = mean(acc)) %>%
  ungroup() %>%
  group_by(experiment, lm, seen_hypernyms, ablation_type, random) %>%
  summarize(
    n = n(),
    sd = sd(acc),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    acc = mean(acc)
  )

exp1_ablation_results <- bind_rows(
  exp1_ablation_results_raw %>% filter(ablation_type != "Full"),
  exp1_ablation_results_raw %>% filter(ablation_type == "Full") %>%
    mutate(ablation_type = "Random"),
  exp1_ablation_results_raw %>% filter(ablation_type == "Full") %>%
    mutate(ablation_type = "Systematic")
) %>%
  ungroup()

exp1_ablation_results %>%
  mutate(
    Rep = case_when(
      random ~ "Random",
      TRUE ~ "Pre-trained"
    )
  ) %>%
  # filter(experiment == "Unseen") %>%
  ggplot(aes(seen_hypernyms, acc, color = lm, shape = lm, fill = lm, linetype = Rep)) +
  geom_point(size = 2)+
  geom_line() +
  geom_ribbon(aes(ymin = acc-conf, ymax=acc+conf), color = NA, alpha = 0.2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.5) +
  # facet_wrap(~ablation_type) +
  facet_grid(ablation_type ~ experiment) +
  scale_x_continuous(limits = c(0,90), breaks = c(0,15,30,45,60,75,90)) +
  scale_y_continuous(limits = c(0.4,1), breaks = c(0.4,0.5,0.6,0.7,0.8,0.9, 1), labels = scales::percent_format(suffix = "")) +
  scale_shape_manual(values = c(21,23)) +
  scale_color_manual(values = c("steelblue", "#e6ab02"), aesthetics=c("fill", "color")) +
  # guides(
  #   color = guide_legend(nrow = 2),
  #   fill = guide_legend(nrow = 2),
  #   # shape = guide_legend("Premise", override.aes = list(alpha = 1), nrow = 1)
  # ) +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    # legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "% of Hypernym Categories Seen in Training",
    y = "Accuracy on unseen images (%)",
    color = "LM",
    fill = "LM",
    shape = "LM",
    linetype = "Representations"
    # color = "Image Encoder"
  )

# 882w x 359h
ggsave("plots/exp1-results.pdf", height = 5.25, width = 10.47, dpi = 300, device=cairo_pdf)

# exp 3?

exp3_results_raw <- bind_rows(
  unseen_shuffled %>%
    select(-overall, -total_illegal_ratio) %>%
    pivot_longer(mammal:`kitchen tool`, names_to = "hypernym", values_to = "acc") %>%
    filter(vision_encoder %in% c("DINOv2"), random == F) %>%
    filter(str_detect(lm, "Qwen")) %>% 
    filter(!is.na(acc)),
  unseen %>%
    select(-overall, -total_illegal_ratio) %>%
    pivot_longer(mammal:condiment, names_to = "hypernym", values_to = "acc") %>%
    filter(vision_encoder %in% c("DINOv2"), random == F) %>%
    filter(str_detect(lm, "Qwen")) %>% 
    filter(!is.na(acc))
) %>%
  mutate(
    shuffled = factor(shuffled, levels = c("Original", "Across-category", "Within-category"))
  )

exp3_results_raw %>%
  count(lm, seed, shuffled)

fit_exp3 <- lmer(
  acc ~ shuffled * lm +
    (1 + shuffled * lm | seed) +
    (1 + shuffled * lm | hypernym),
  data = exp3_results_raw %>%
    filter(ablation_type == "Full")
)

summary(fit_exp3)

# significant differences between none and across (diff = 0.30, t = 14.9, p < 0.001) 
# and between none and within (diff= 0.04,t = 3.9, p < 0.05)

exp3_results_raw %>%
  filter(ablation_type == "Full") %>%
  group_by(lm, shuffled, hypernym) %>%
  summarize(acc = mean(acc)) %>%
  ungroup() %>%
  group_by(lm, shuffled) %>%
  summarize(
    n = n(),
    sd = sd(acc),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    acc = mean(acc)
  ) %>%
  ggplot(aes(lm, acc, color = shuffled, shape = shuffled)) +
  geom_point(size = 2, position = position_dodge(0.5))+
  geom_linerange(aes(ymin = acc-conf, ymax = acc+conf), position = position_dodge(0.5)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.5) +
  scale_y_continuous(limits = c(0.4,1), breaks = c(0.4,0.5,0.6,0.7,0.8,0.9, 1), labels = scales::percent_format(suffix = ""))


exp3_ablation_results <- bind_rows(
  exp3_results_raw %>% filter(ablation_type != "Full"),
  exp3_results_raw %>% filter(ablation_type == "Full") %>%
    mutate(ablation_type = "Random"),
  exp3_results_raw %>% filter(ablation_type == "Full") %>%
    mutate(ablation_type = "Systematic")
) %>%
  ungroup()

exp3_ablation_results %>% View()

exp3_ablation_results %>%
  filter(ablation_type=="Random") %>%
  group_by(lm, seen_hypernyms, ablation_type, shuffled, hypernym) %>%
  summarize(acc = mean(acc)) %>%
  ungroup() %>%
  group_by(lm,seen_hypernyms, ablation_type,shuffled) %>%
  summarize(
    n = n(),
    sd = sd(acc),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    acc = mean(acc)
  ) %>%
  ggplot(aes(seen_hypernyms, acc, color = shuffled, fill = shuffled, shape = shuffled)) +
  geom_point(size = 2)+
  geom_line() +
  geom_ribbon(aes(ymin = acc-conf, ymax=acc+conf), color = NA, alpha = 0.2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.5) +
  # facet_grid(lm ~ ablation_type) +
  facet_wrap(~lm) +
  scale_y_continuous(limits = c(0.4,1), breaks = c(0.4,0.5,0.6,0.7,0.8,0.9, 1), labels = scales::percent_format(suffix = "")) +
  scale_x_continuous(limits = c(0,90), breaks = c(0,15,30,45,60,75,90)) +
  scale_shape_manual(values = c(21,23,24)) +
  scale_color_manual(values = c("#7570b3", "#1b9e77", "#d95f02"), aesthetics=c("fill", "color")) +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    legend.box.margin = margin(0,0,0,-30),
    axis.title = element_text(size = 16),
    legend.title = element_text(size=16)
  ) +
  labs(
    x = "% of Hypernym Categories Seen in Training",
    y = "Accuracy on\nunseen images (%)",
    color = "Shuffling",
    fill = "Shuffling",
    shape = "Shuffling"
    # color = "Image Encoder"
  )

ggsave("plots/counterfactual-shuffling-results-generalization.pdf", height = 4.35, width = 6.44, dpi=300, device=cairo_pdf)

