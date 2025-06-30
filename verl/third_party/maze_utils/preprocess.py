"""
Preprocess the data to make it ready for the reward function.

1. jsonl file
2.
data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
3. All the data to parquet file
"""

import json
import os
import argparse
import pandas as pd


def preprocess_data(data_path: str, output_dir: str):
    data_source = data_path.split("/")[-1].split(".")[0]
    all_data = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            full_text = f"<|bos|> {data['text']} <|eos|>"
            tokens = full_text.split()
            reasoning_idx = tokens.index("reasoning")
            prompt_tokens = tokens[: reasoning_idx + 1]
            completion_tokens = tokens[reasoning_idx + 1:]
            prompt = " ".join(prompt_tokens)
            completion = " ".join(completion_tokens)
            spec = prompt.replace("<|bos|>", "").replace(" query ", "").replace(" reasoning", "").strip()
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "planning",
                "reward_model": {"style": "rule", "ground_truth": completion},
                "extra_info": {"spec": spec, "maze_size": (30, 30)},
            }
            all_data.append(data)
    # Save all the data to parquet file
    df = pd.DataFrame(all_data)
    df.to_parquet(os.path.join(output_dir, f"{data_source}.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    train_path = f"{args.data_path}/train_{args.data_set}.jsonl"
    eval_path = f"{args.data_path}/test_{args.data_set}.jsonl"
    preprocess_data(train_path, args.output_dir)
    preprocess_data(eval_path, args.output_dir)
    
    
