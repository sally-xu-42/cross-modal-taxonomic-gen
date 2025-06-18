from vlm_semantics.src.x_dataloader import CLEVRDataset

train_dataset = CLEVRDataset(
    questions_json=Path("CLEVR_train_questions.json"),
    image_dir=Path("images/train"),
    image_transform=some_transform,
    tokenizer=some_tokenizer,
    split="train",
)

val_dataset = CLEVRDataset(
    questions_json=Path("CLEVR_val_questions.json"),
    image_dir=Path("images/val"),
    image_transform=some_transform,
    tokenizer=some_tokenizer,
    split="val",
)

# Then feed these datasets into your PRISMATIC training loop, trainer, or pipeline.
trainer = Trainer(
    model=some_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()