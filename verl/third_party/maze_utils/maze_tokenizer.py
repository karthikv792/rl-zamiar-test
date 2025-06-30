# TODO fix this so it actually changes RoPE properly! (see Karthik stuff)
# Create and save the custom tokenizer
from rich import print
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from typing import List
import os
import argparse

def return_vocabulary(width: int, height: int):
    """Generates the custom vocabulary list."""
    vocabulary: List[str] = []
    # Cell identifiers (c0 to c(width*height - 1))
    vocabulary += [f"c{i}" for i in range(width * height)]
    # Coordinates (0 to max(width, height)-1)
    vocabulary += [str(i) for i in range(max(width, height))]
    # Action/marker tokens
    vocabulary += ["start", "goal", "wall", "create", "close", "plan", "query", "reasoning", "solution", "end"]
    return vocabulary

def create_custom_tokenizer(
    width: int,  # Maze width
    height: int,  # Maze height
    tokenizer_save_dir: str,
    special_tokens: List[str] = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]
):
    """Creates and saves a custom tokenizer with the specified parameters."""
    # 1. Generate Vocabulary
    custom_vocab = return_vocabulary(width, height)
    full_vocab_list = custom_vocab + special_tokens
    vocab_map = {token: i for i, token in enumerate(full_vocab_list)}
    print(f"Total vocabulary size: {len(vocab_map)}")
    print(f"First 10 vocab items: {list(vocab_map.items())[:10]}")
    print(f"Last 10 vocab items: {list(vocab_map.items())[-10:]}")

    # 2. Initialize and Train Tokenizer (using WordLevel requires a map)
    tokenizer = Tokenizer(WordLevel(vocab=vocab_map, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace() # type: ignore
    num_added_toks = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added_toks} special tokens.")

    if "<|pad|>" in special_tokens:
        pad_token_id = vocab_map["<|pad|>"]
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token="<|pad|>")

    # 3. Save the Tokenizer
    os.makedirs(tokenizer_save_dir, exist_ok=True)
    tokenizer_save_path = os.path.join(tokenizer_save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_save_path)
    print(f"Custom tokenizer saved to: {tokenizer_save_path}")

    # 4. Save tokenizer in HuggingFace format
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_save_path,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )
    
    # Add chat template for maze-solving conversations
    maze_chat_template = "{{ messages }}"
    
    hf_tokenizer.chat_template = maze_chat_template
    hf_tokenizer.save_pretrained(tokenizer_save_dir)
    print(f"HuggingFace tokenizer saved to: {tokenizer_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a custom tokenizer for maze tasks")
    parser.add_argument("--width", type=int, default=30, help="Width of the maze")
    parser.add_argument("--height", type=int, default=30, help="Height of the maze")
    parser.add_argument(
        "--tokenizer_save_dir",
        type=str,
        default="./data/tokenizer",
        help="Directory to save the tokenizer"
    )
    args = parser.parse_args()

    create_custom_tokenizer(
        width=args.width,
        height=args.height,
        tokenizer_save_dir=args.tokenizer_save_dir
    )
