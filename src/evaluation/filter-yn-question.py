import argparse
import csv
import json
import pathlib
import torch

from minicons import scorer
from torch.utils.data import DataLoader
from tqdm import tqdm


STIMULI_PATH = "data/llm-backbone-exp-data/yn_questions.jsonl"


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def write_csv(data, path, header=None):
    with open(path, "w") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)


def main(args):
    model = args.model
    model_name = args.model.replace("/", "_")
    output_dir = args.output_dir

    # question_scorer -- a large LM.
    lm = scorer.IncrementalLMScorer(model, device=args.device)

    stimuli = read_jsonl(STIMULI_PATH)
    stimuli_formatted = [
        {"idx": i, "question": entry["question"]} for i, entry in enumerate(stimuli)
    ]

    batches = DataLoader(stimuli_formatted, batch_size=args.batch_size)

    results = []
    for batch in tqdm(batches, desc="Scoring YN questions"):
        idx = [int(x) for x in batch["idx"]]
        questions = batch["question"]
        scores = lm.sequence_score(questions)
        for idx, score in zip(idx, scores):
            results.append((idx, score))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_csv(
        results,
        path=f"{output_dir}/{model_name}.csv",
        header=["idx", "score"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/llm-backbone-yn-filtering/",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )

    args = parser.parse_args()
    main(args)
