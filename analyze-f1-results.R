library(tidyverse)
library(lmerTest)
library(ggbeeswarm)
library(ggdist)

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

hyp_random <- read_csv("results/main-results/macro-f1s/all_hyp_guess_f1_scores.csv") %>%
  mutate(
    shuffled = case_when(
      str_detect(run_name, "local_shuffled") ~ "Within-category",
      !str_detect(run_name, "_shuffled") ~ "Original",
      TRUE ~ "Across-category"
    ),
    ablation_type = case_when(
      str_detect(run_name, "\\dcat") ~ "Systematic",
      str_detect(run_name, "\\dpct") ~ "Random",
      TRUE ~ "Full"
    ),
    ablation_amt = case_when(
      str_detect(run_name, "(ablated|trained)_\\d{2}cat") ~ as.numeric(str_extract(run_name, "(?<=(ablated|trained)_)(.*)(?=cat)")),
      str_detect(run_name, "(ablated|trained)_\\d{2}pct") ~ as.numeric(str_extract(run_name, "(?<=(ablated|trained)_)(.*)(?=pct)")),
      TRUE ~ 100
    ),
    seen_hypernyms = case_when(
      str_detect(run_name, "(ablated|trained)_\\d{2}cat") ~ 100 * ((53-ablation_amt)/53),
      str_detect(run_name, "(ablated|trained)_\\d{2}pct") ~ 100-ablation_amt,
      TRUE ~ 100-ablation_amt
    ),
    test = str_detect(run_name, "(ablated|trained)_\\d{2}pct"),
    experiment = case_when(
      str_detect(run_name, "trained") ~ "Seen Hypernyms",
      TRUE ~ "Held-Out Hypernyms"
    )
  )

hyp_random %>%
  count(run_name, ablation_amt, seen_hypernyms, test, experiment) %>% View()

seen_hypernyms <- hyp_random %>% distinct(seen_hypernyms) %>% pull(seen_hypernyms)
ablation_type <- c("Random", "Systematic")

leaf_random <- read_csv("results/main-results/macro-f1s/all_leaf_guess_f1_scores.csv") %>%
  filter(run_name == "test") %>%
  mutate(
    shuffled = "Original",
    experiment = "Leaves"
  ) 

leaf_random_chance_f1 <- crossing(
  leaf_random, 
  seen_hypernyms = seen_hypernyms, 
  ablation_type = ablation_type
) %>%
  pivot_longer(aardvark:zucchini, names_to = "category", values_to = "f1") %>%
  group_by(shuffled, experiment, seen_hypernyms, ablation_type) %>%
  summarize(
    n = n(),
    sd = sd(f1),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    acc = mean(f1)
  ) %>%
  ungroup()

chance_f1_raw <- hyp_random %>%
  pivot_longer(animal:weapon, names_to = "category", values_to = "f1") %>%
  group_by(shuffled, experiment, seen_hypernyms, ablation_type) %>%
  summarize(
    n = n(),
    sd = sd(f1),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    acc = mean(f1)
  ) %>%
  ungroup()

chance_f1 <- bind_rows(
  bind_rows(
    chance_f1_raw %>% filter(ablation_type != "Full"),
    chance_f1_raw %>% filter(ablation_type == "Full") %>%
      mutate(ablation_type = "Random"),
    chance_f1_raw %>% filter(ablation_type == "Full") %>%
      mutate(ablation_type = "Systematic")
  ) %>%
    ungroup(),
  leaf_random_chance_f1
)

# leaf_random

unseen_counts <- read_csv("results/main-results/all_unseen_counts.csv")
leaf_counts <- read_csv("results/main-results/all_leaf_counts.csv")
seen_counts <- read_csv("results/main-results/all_seen_counts.csv")


leaf <- read_results("results/main-results/macro-f1s/all_leaf_f1_scores.csv") %>% 
  distinct() %>%
  filter(shuffled == "Original")
seen <- read_results("results/main-results/macro-f1s/all_seen_f1_scores.csv") %>% 
  distinct() %>%
  filter(shuffled == "Original")
unseen <- read_results("results/main-results/macro-f1s/all_unseen_f1_scores.csv") %>% 
  distinct() %>%
  filter(shuffled == "Original")

unseen_shuffled <- read_results("results/main-results/macro-f1s/all_unseen_f1_scores.csv") %>%
  distinct() %>%
  filter(ablation_type %in% c("Random", "Full"), vision_encoder=="DINOv2", random==F)

dino_vs_siglip <- unseen %>%
  pivot_longer(animal:weapon, names_to = "hypernym", values_to = "f1") %>%
  filter(vision_encoder %in% c("SigLIP", "DINOv2"), ablation_type == "Full", random == F) %>%
  filter(str_detect(lm, "Qwen"))

fit_dinosiglip <- lmer(
  f1 ~ vision_encoder * lm + 
    (1 + vision_encoder * lm | seed) + 
    (1 + vision_encoder * lm | hypernym),
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

dino_vs_siglip %>%
  # group_by(lm, vision_encoder, hypernym) %>%
  # summarize(f1 = mean(f1)) %>%
  # ungroup() %>%
  group_by(lm, vision_encoder) %>%
  summarize(
    n = n(),
    sd = sd(f1),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    f1 = mean(f1)
  ) %>% 
  ggplot(aes(lm, f1, color = vision_encoder, fill = vision_encoder, shape = vision_encoder)) +
  geom_point(size = 2, position = position_dodge(0.5))+
  geom_linerange(aes(ymin = f1-conf, ymax = f1+conf), position = position_dodge(0.5)) +
  geom_hline(yintercept = 0.471, linetype = "dashed", linewidth = 0.5) +
  scale_y_continuous(limits = c(0.4, 1), labels = scales::percent_format(suffix = ""), breaks = c(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) +
  scale_shape_manual(values = c(21,23)) +
  theme_bw(base_size = 18, base_family = "Times") +
  # theme_classic(base_size = 18, base_family = "Times") +
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
    y = "F1 on unseen images",
    color = "Image Encoder",
    fill = "Image Encoder",
    shape = "Image Encoder"
  )

ggsave("plots/dino-vs-siglip.pdf", width = 4.45, height = 3.6, dpi=300, device=cairo_pdf)


# exp1_ablation_results_raw %>% count(ablation_type)

exp1_ablation_results_raw <- bind_rows(
  unseen %>%
    pivot_longer(animal:weapon, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Held-Out Hypernyms") %>%
    inner_join(
      unseen_counts %>%
        pivot_longer(animal:weapon, names_to = "category", values_to = "count")
    ),
  seen %>%
    pivot_longer(animal:weapon, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Seen Hypernyms") %>%
    inner_join(
      seen_counts %>%
        pivot_longer(animal:weapon, names_to = "category", values_to = "count")
    ),
  leaf %>%
    pivot_longer(aardvark:zucchini, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Leaves") %>%
    inner_join(
      leaf_counts %>%
        pivot_longer(aardvark:zucchini, names_to = "category", values_to = "count")
    )
) %>%
  mutate(
    experiment = factor(experiment, levels = c("Leaves", "Seen Hypernyms", "Held-Out Hypernyms"))
  ) %>%
  filter(vision_encoder %in% c("DINOv2")) %>%
  filter(str_detect(lm, "Qwen")) %>% 
  filter(!is.na(acc)) %>%
  group_by(experiment, lm, seen_hypernyms, ablation_type, category, random) %>%
  # summarize(n = n(), acc = mean(acc), count = mean(count)) %>%
  # ungroup() %>%
  group_by(experiment, lm, seen_hypernyms, ablation_type, random) %>%
  summarize(
    total = sum(count),
    weighted = sum(count * acc)/sum(count),
    weighted_sd = sqrt(sum(count * ((acc - (sum(count * acc)/sum(count)))^2))/(sum(count) - 1)),
    weighted_conf = 1.96 * weighted_sd/sqrt(sum(count)),
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

joint_seen_ablation <- exp1_ablation_results %>%
  distinct(seen_hypernyms, ablation_type)

exp1_chance <- chance_f1 %>% 
  filter(shuffled == "Original") %>%
  inner_join(joint_seen_ablation) %>% 
  mutate(
    deprivation=100-seen_hypernyms,
    ablation_type = glue::glue("{ablation_type} Ablation"),
    experiment = factor(experiment, levels = c("Leaves", "Seen Hypernyms", "Held-Out Hypernyms"))
  )


unseen_categorywise <- unseen_shuffled %>%
  pivot_longer(animal:weapon, names_to = "category", values_to = "acc") %>%
  mutate(experiment = "Held-Out Hypernyms") %>%
  inner_join(
    unseen_counts %>%
      pivot_longer(animal:weapon, names_to = "category", values_to = "count")
  ) %>%
  filter(vision_encoder %in% c("DINOv2")) %>%
  filter(str_detect(lm, "Qwen")) %>%
  filter(seen_hypernyms==0) %>%
  filter(
    # experiment == "Held-Out Hypernyms",
    # lm == "Qwen3-0.6B",
    # seed == 42
  ) %>%
  mutate(
    model = case_when(
      random == FALSE ~ "LM",
      TRUE ~ "Random"
    )
  ) %>%
  filter(random == FALSE) %>%
  select(shuffled, model = lm, seed, category, acc) %>%
  mutate(
   category = case_when(
     category == "musical" ~ "musical instrument",
     category == "school supply" ~ "school supply item",
     TRUE ~ category
    ) 
  )

reg_data <- unseen_categorywise %>%
  filter(shuffled == "Original") %>%
  inner_join(coherence_backbone %>% filter(shuffle_type == "original")) %>%
  mutate(
    model = factor(model),
    category = factor(category),
    seed = factor(seed)
  )

fit1 <- lmer(acc ~ backbone_acc + avg_cosine + (backbone_acc + avg_cosine || category) + (1|seed) + (1|model), data = reg_data)

summary(fit1)

reg_data %>% 
  group_by(model, seed) %>% 
  nest() %>%
  mutate(
    cor = map(data, function(x) {
      cor.test(x$acc, x$avg_cosine, method = "spearman") %>%
        broom::tidy()
    })
  ) %>%
  unnest(cor)

reg_data %>%
  ggplot(aes(avg_cosine, acc)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_grid(model ~ seed) +
  labs(
    x = "Visual Coherence\n(avg. pairwise cosine between images ofcategory members)",
    y = "Macro F1 on\nHeld-out Hypernyms (N = 53)"
  )


reg_data_within <- unseen_categorywise %>%
  filter(str_detect(shuffled, "Within")) %>%
  inner_join(coherence_backbone %>% filter(shuffle_type == "local_shuffled")) %>%
  mutate(
    model = factor(model),
    category = factor(category),
    seed = factor(seed)
  )

fit1_within <- lmer(acc ~ backbone_acc + avg_cosine + (backbone_acc + avg_cosine || category) + (1|seed) + (1|model), data = reg_data_within)

summary(fit1_within)

reg_data_across <- unseen_categorywise %>%
  filter(str_detect(shuffled, "Across")) %>%
  inner_join(coherence_backbone %>% filter(shuffle_type == "shuffled")) %>%
  mutate(
    model = factor(model),
    category = factor(category),
    seed = factor(seed)
  )

fit1_across <- lmer(acc ~ backbone_acc + avg_cosine + (backbone_acc * avg_cosine || category) + (1|seed) + (1|model), data = reg_data_across)

summary(fit1_across)

reg_data_across %>%
  ggplot(aes(avg_cosine, acc)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_grid(model ~ seed)

reg_data_within %>% 
  group_by(model, seed) %>% 
  nest() %>%
  mutate(
    cor = map(data, function(x) {
      cor.test(x$acc, x$avg_cosine, method = "spearman") %>%
        broom::tidy()
    })
  ) %>%
  unnest(cor)


unseen %>%
  pivot_longer(animal:weapon, names_to = "category", values_to = "acc") %>%
  mutate(experiment = "Held-Out Hypernyms") %>%
  inner_join(
    unseen_counts %>%
      pivot_longer(animal:weapon, names_to = "category", values_to = "count")
  ) %>%
  filter(vision_encoder %in% c("DINOv2")) %>%
  filter(str_detect(lm, "Qwen")) %>%
  filter(seen_hypernyms==0) %>%
  filter(
    # experiment == "Held-Out Hypernyms",
    lm == "Qwen3-0.6B",
    seed == 42
  ) %>%
  mutate(
    model = case_when(
      random == FALSE ~ "LM",
      TRUE ~ "Random"
    )
  ) %>% 
  ggplot(aes(model, acc)) +
  # geom_point(color = "white") +
  # geom_quasirandom(width = 0.1, alpha = 0.6, color="white", shape = 21, fill = "#bf5700") +
  geom_quasirandom(width = 0.1, alpha = 0.6, color="white", shape = 21, fill = "mediumpurple4") +
  stat_summary(fun.data = "mean_cl_normal", size = 0.3) +
  # stat_boxplot() +
  # stat_interval(.width = c(0.05, 0.2, 0.5, 0.8, 0.95), size = 10) +
  # stat_interval() +
  # stat_gradientinterval(width = 0.2, fill = "goldenrod1") +
  # stat_gradientinterval(width = 0.2, fill = "#bf5700") +
  # stat_dots() +
  geom_hline(yintercept = 0.4, linetype="dashed") +
  # scale_fill_manual(values = scales::brewer_pal()(3)[-1], aesthetics = "slab_fill") +
  # scale_color_brewer(palette = "Blues", aesthetics = c("color", "fill")) +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  theme_classic(base_size = 17, base_family = "DM Sans") +
  theme(
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Model",
    y = "Generalization"
  )


talk_plot <- exp1_ablation_results %>%
  mutate(
    Rep = case_when(
      random ~ "Random",
      TRUE ~ "Pre-trained"
    ),
    deprivation = 100-seen_hypernyms,
    ablation_type = glue::glue("{ablation_type} Ablation"),
  ) %>%
  filter(
    # seen_hypernyms==0.0,
    ablation_type == "Random Ablation", 
    experiment == "Held-Out Hypernyms",
    lm == "Qwen3-0.6B"
  ) 

talk_plot %>%
  filter(seen_hypernyms == 0.0)

exp1_ablation_results %>%
  mutate(
    Rep = case_when(
      random ~ "Random",
      TRUE ~ "Pre-trained"
    ),
    deprivation = 100-seen_hypernyms,
    ablation_type = glue::glue("{ablation_type} Ablation"),
  ) %>%
  # filter(experiment == "Unseen") %>%
  ggplot(aes(group = interaction(lm, Rep))) +
  geom_point(aes(deprivation, acc, color = lm, shape = lm, fill = lm), size = 2)+
  geom_line(aes(deprivation, acc, color = lm, linetype = Rep)) +
  geom_ribbon(aes(deprivation, acc, ymin = acc-conf, ymax=acc+conf, fill = lm), color = NA, alpha = 0.2) +
  # geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.5) +
  # facet_wrap(~ablation_type) +
  facet_grid(ablation_type ~ experiment) +
  scale_x_continuous(limits = c(0,100)) +
  # scale_y_continuous(limits = c(0.3,1), breaks = c(0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1), labels = scales::percent_format(suffix = "")) +
  scale_y_continuous(limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8,1), labels = scales::percent_format(suffix = "")) +
  scale_shape_manual(values = c(21,23)) +
  scale_color_manual(values = c("steelblue", "#e6ab02"), aesthetics=c("fill", "color")) +
  scale_linetype_manual(values = c("solid", "dotted")) +
  geom_line(data=exp1_chance, aes(x = deprivation, y = acc, group = 1), color = "black", linetype = "dashed") +
  geom_point(data=exp1_chance, aes(x = deprivation, y = acc, group = 1), color = "black", fill = "black", size = 1) +
  geom_ribbon(data=exp1_chance, aes(x = deprivation, y = acc, ymin = acc-conf, ymax=acc+conf, group = 1), color = NA, fill = "black", alpha = 0.2) +
  # guides(
  #   color = guide_legend(nrow = 2),
  #   fill = guide_legend(nrow = 2),
  #   # shape = guide_legend("Premise", override.aes = list(alpha = 1), nrow = 1)
  # ) +
  theme_bw(base_size = 18, base_family = "Times") +
  # theme_classic(base_size = 18, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    # legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "% of Image-Hypernym pairs ablated from training",
    y = "Macro F1 on unseen images",
    color = "LM Backbone",
    fill = "LM Backbone",
    shape = "LM Backbone",
    linetype = "Representations"
    # color = "Image Encoder"
  )

ggsave("plots/main-exp-results.pdf", height = 5.25, width = 10.47, dpi = 300, device=cairo_pdf)

exp3_chance <- chance_f1 %>% 
  filter(ablation_type == "Random", experiment=="Held-Out Hypernyms") %>%
  inner_join(joint_seen_ablation) %>% 
  mutate(
    deprivation=100-seen_hypernyms,
    # ablation_type = glue::glue("{ablation_type} Ablation"),
    # experiment = factor(experiment, levels = c("Leaves", "Seen Hypernyms", "Held-Out Hypernyms"))
  )

unseen_shuffled %>%
  pivot_longer(animal:weapon, names_to = "category", values_to = "acc") %>%
  mutate(experiment = "Held out Hypernyms") %>%
  inner_join(
    unseen_counts %>%
      pivot_longer(animal:weapon, names_to = "category", values_to = "count")
  ) %>%
  mutate(
    shuffled = factor(shuffled, levels = c("Original", "Across-category", "Within-category"))
  ) %>%
  # group_by(lm, seen_hypernyms, ablation_type, shuffled, hypernym) %>%
  # summarize(acc = mean(acc)) %>%
  # ungroup() %>%
  group_by(lm,seen_hypernyms, ablation_type,shuffled) %>%
  summarize(
    n = n(),
    sd = sd(acc),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    acc = mean(acc)
  ) %>%
  ungroup() %>%
  mutate(
    deprivation = 100-seen_hypernyms
  ) %>%
  ggplot(aes(deprivation, acc, color = shuffled, fill = shuffled, shape = shuffled)) +
  geom_point(size = 2)+
  geom_line() +
  geom_ribbon(aes(ymin = acc-conf, ymax=acc+conf), color = NA, alpha = 0.2) +
  geom_point(data = exp3_chance, color="black", fill="black", size = 1, show.legend = F) +
  geom_line(data = exp3_chance, color="black", linewidth = 0.5, linetype = "dashed", show.legend = F) +
  geom_ribbon(data = exp3_chance, aes(ymin = acc-conf, ymax=acc+conf), fill = "black", color = NA, alpha = 0.05, show.legend = F) +
  # geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.5) +
  # facet_grid(lm ~ ablation_type) +
  facet_wrap(~lm) +
  # scale_y_continuous(limits = c(0.4,1), breaks = c(0.4,0.5,0.6,0.7,0.8,0.9, 1), labels = scales::percent_format(suffix = "")) +
  scale_y_continuous(limits = c(0,1), breaks = c(0, 0.2, 0.4,0.6,0.8, 1), labels = scales::percent_format(suffix = "")) +
  scale_x_continuous(limits = c(0,100)) +
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
    x = "% of Image-Hypernym pairs ablated from training",
    y = "Macro F1/category\non unseen images",
    color = "Shuffle",
    fill = "Shuffle",
    shape = "Shuffle"
    # color = "Image Encoder"
  )

ggsave("plots/counterfactual-shuffling-results-generalization.pdf", height = 4.35, width = 6.44, dpi=300, device=cairo_pdf)



exp1_llama_raw <- bind_rows(
  unseen %>%
    pivot_longer(animal:weapon, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Held-Out Hypernyms"),
  seen %>%
    pivot_longer(animal:weapon, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Seen Hypernyms"),
  leaf %>%
    pivot_longer(aardvark:zucchini, names_to = "category", values_to = "acc") %>%
    mutate(experiment = "Leaves")
) %>%
  mutate(
    experiment = factor(experiment, levels = c("Leaves", "Seen Hypernyms", "Held-Out Hypernyms"))
  ) %>%
  filter(vision_encoder %in% c("DINOv2")) %>%
  filter(str_detect(lm, "Llama")) %>% 
  filter(!is.na(acc)) %>%
  group_by(experiment, lm, seen_hypernyms, ablation_type, category, random) %>%
  group_by(experiment, lm, seen_hypernyms, ablation_type, random) %>%
  summarize(
    n = n(),
    sd = sd(acc),
    conf = qt(0.05/2, n - 1, lower.tail = FALSE) * sd/sqrt(n),
    acc = mean(acc)
  )

exp1_llama <- bind_rows(
  exp1_llama_raw %>% filter(ablation_type != "Full"),
  exp1_llama_raw %>% filter(ablation_type == "Full") %>%
    mutate(ablation_type = "Random"),
  exp1_llama_raw %>% filter(ablation_type == "Full") %>%
    mutate(ablation_type = "Systematic")
) %>%
  ungroup()

exp1_llama %>%
  mutate(
    Rep = case_when(
      random ~ "Random",
      TRUE ~ "Pre-trained"
    ),
    deprivation = 100-seen_hypernyms,
    ablation_type = glue::glue("{ablation_type} Ablation"),
  ) %>%
  # filter(experiment == "Unseen") %>%
  ggplot(aes(group = interaction(lm, Rep))) +
  geom_point(aes(deprivation, acc, color = lm, shape = lm, fill = lm), size = 2)+
  geom_line(aes(deprivation, acc, color = lm, linetype = Rep)) +
  geom_ribbon(aes(deprivation, acc, ymin = acc-conf, ymax=acc+conf, fill = lm), color = NA, alpha = 0.2) +
  # geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.5) +
  # facet_wrap(~ablation_type) +
  facet_grid(ablation_type ~ experiment) +
  scale_x_continuous(limits = c(0,100)) +
  scale_y_continuous(limits = c(0.4,1), breaks = c(0.4,0.5,0.6,0.7,0.8,0.9, 1), labels = scales::percent_format(suffix = "")) +
  scale_shape_manual(values = c(21,23)) +
  scale_color_manual(values = c("steelblue", "#e6ab02"), aesthetics=c("fill", "color")) +
  scale_linetype_manual(values = c("solid", "dotted")) +
  geom_line(data=exp1_chance, aes(x = deprivation, y = acc, group = 1), color = "black", linetype = "dashed") +
  geom_point(data=exp1_chance, aes(x = deprivation, y = acc, group = 1), color = "black", fill = "black", size = 1) +
  geom_ribbon(data=exp1_chance, aes(x = deprivation, y = acc, ymin = acc-conf, ymax=acc+conf, group = 1), color = NA, fill = "black", alpha = 0.2) +
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
    x = "% of Image-Hypernym pairs ablated from training",
    y = "F1 on unseen images",
    color = "LM Backbone",
    fill = "LM Backbone",
    shape = "LM Backbone",
    linetype = "Representations"
    # color = "Image Encoder"
  )
