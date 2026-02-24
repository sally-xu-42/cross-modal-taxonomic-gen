library(tidyverse)

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

read_results("~/Downloads/cat_leaf_f1_scores.csv") %>% 
  distinct() %>%
  pivot_longer(`air conditioner`:zucchini, names_to = "hypernym", values_to = "f1") %>%
  group_by(run_name) %>%
  summarise(
    f1 = mean(f1)
  )

read_results("~/Downloads/cat_unseen_f1_scores (1).csv") %>% 
  distinct() %>%
  pivot_longer(animal:weapon, names_to = "hypernym", values_to = "f1") %>%
  group_by(run_name) %>%
  summarise(
    n = n(),
    f1 = mean(f1)
  )
