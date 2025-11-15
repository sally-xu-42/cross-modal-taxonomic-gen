import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_question(concept: str, category: str) -> str:
    """Generate question in format: Is a/an X a/an Y?"""
    def get_article(word):
        return "an" if word[0].lower() in 'aeiou' else "a"
    
    obj = concept.replace('_', ' ')
    return f"Is {get_article(obj)} {obj} {get_article(category)} {category}? Answer only in yes or no."


def scoring(model, tokenizer, question: str, label: str):
    """Score candidates using yes/no."""
    messages = [
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(messages, 
                                         tokenize=False, 
                                         add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.0,
            do_sample=False
        )
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print("content:", content)
        if label.lower() in content.lower():
            return content.lower(), True
        else:
            return content.lower(), False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--input_path", type=str, default="./data/hypernymy_THINGS/things-hyp.jsonl") # category-pairs.csv
    parser.add_argument("--output_path", type=str, default="./lm_results/qwen_3_4b_inst_res.csv")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    hf_token = os.environ.get(".hf_token", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=hf_token)

    df = pd.read_json(args.input_path, lines=True)
    print(df.head())
    
    # Evaluate
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        concept, category, label = row['concept'], row['hypernym'], row['label']
        question = make_question(concept, category)
        print(f"Q: {question} | A: {label}")
        predicted, correct = scoring(model, tokenizer, question, label)
        
        results.append({
            "concept": concept,
            "category": category,
            "question": question,
            "label": label,
            "predicted": predicted,
            "correct": correct,
        })
    
    # Calculate accuracy
    overall = sum(r["correct"] for r in results) / len(results) if results else 0
    
    # Per-category accuracy
    category_acc = {}
    df_results = pd.DataFrame(results)
    for cat in df_results['category'].unique():
        cat_results = df_results[df_results['category'] == cat]
        category_acc[cat] = cat_results['correct'].mean()
    
    print(f"\nOverall Accuracy: {overall * 100:.1f}%")
    illegal = 0
    for predicted in df_results['predicted']:
        if predicted.lower() != "yes" and predicted.lower() != "no":
            illegal += 1
            print(f"Unrecognized answer: {predicted}")
    print(f"\nIllegal answer rate: {illegal / len(df_results) * 100:.1f}%")

    # print(f"\nTop 5 Categories by Accuracy:")
    # for cat, acc in sorted(category_acc.items(), key=lambda x: x[1], reverse=True)[:5]:
    #     print(f"  {cat}: {acc * 100:.1f}%")
    
    # Save results
    output = {
        "accuracy": f"{overall*100:.1f}",
        "category_accuracy": category_acc,
        # "results": results
    }
    summary_path = args.output_path.replace(".csv", "_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(json.dumps(output))
    df_results.to_csv(args.output_path, index=False)
    print(f"\nSaved to {args.output_path} and {summary_path}")