import argparse
import csv
import json
import pathlib
import torch

from minicons import scorer
from torch.utils.data import DataLoader
from tqdm import tqdm


STIMULI_PATH = "data/llm-backbone-exp-data/yn_questions.jsonl"

OPTIONS = ["Yes", "No", "yes", "no"]


def p_yes(probs):
    alls = torch.tensor(probs).sum(1)
    yeses = [[p[0], p[2]] for p in probs]
    return (torch.tensor(yeses).sum(1) / alls).tolist()


def rank_yes(ranks):
    yes_ranks = [min(p[0], p[2]) for p in ranks]
    return yes_ranks


def rank_no(ranks):
    no_ranks = [min(p[1], p[3]) for p in ranks]
    return no_ranks


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def write_csv(data, path, header=None):
    with open(path, "w") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)


def chat_template(lm, sentence):
    message = [{"role": "user", "content": f"{sentence}"}]
    return lm.tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main(args):
    model = args.model
    model_name = args.model.replace("/", "_")
    output_dir = args.output_dir

    lm = scorer.IncrementalLMScorer(model, device=args.device)

    stimuli = read_jsonl(STIMULI_PATH)
    stimuli_formatted = [
        {"idx": i, "question": chat_template(lm, entry["question"])}
        for i, entry in enumerate(stimuli)
    ]

    print(stimuli_formatted[1])

    results = []

    batches = DataLoader(stimuli_formatted, batch_size=args.batch_size)
    for batch in tqdm(batches):
        idx = [int(x) for x in batch["idx"]]
        inputs = batch["question"]

        dist = lm.next_word_distribution(inputs)
        probs, ranks = lm.query(dist, [OPTIONS] * len(inputs))
        yes_probs = p_yes(probs)
        yes_ranks = rank_yes(ranks)
        no_ranks = rank_no(ranks)

        for p, r, n, i in zip(yes_probs, yes_ranks, no_ranks, idx):
            results.append((i, p, r, n))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_csv(
        results,
        path=f"{output_dir}/{model_name}.csv",
        header=["idx", "p_yes", "rank_yes", "rank_no"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="kanishka_res/llm-backbone-yn/",
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
