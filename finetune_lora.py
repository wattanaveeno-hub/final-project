"""
Fine-tune a Thai LLM (e.g. Typhoon-7B) on the Thai Education QA dataset
using PEFT + LoRA. This script is meant to run on a machine with a CUDA GPU
(A100, RTX 4090, or similar with at least 24 GB VRAM).

Reference: Do, Nguyen & Dam (2025), arXiv:2501.15022

Usage:
    python finetune_lora.py \\
        --model_name scb10x/typhoon-7b \\
        --data_file ../data/thai_edu_qa.jsonl \\
        --output_dir ./checkpoints/typhoon-thai-edu \\
        --num_epochs 10 \\
        --batch_size 4 \\
        --learning_rate 2e-4

Dependencies (install separately on the GPU machine):
    pip install transformers>=4.40 peft>=0.10 datasets accelerate bitsandbytes
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """### คำสั่ง (Instruction):
ตอบคำถามต่อไปนี้โดยอ้างอิงจากบริบทที่ให้

### บริบท (Context):
{context}

### คำถาม (Question):
{question}

### คำตอบ (Answer):
{answer}"""


def clean_thai(text: str) -> str:
    """Same preprocessing as the notebook."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_example(ex):
    return {
        "text": PROMPT_TEMPLATE.format(
            context=clean_thai(ex["context"]),
            question=clean_thai(ex["question"]),
            answer=clean_thai(ex["answer"]),
        )
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="scb10x/typhoon-7b",
                        help="HF hub name or local path of base model")
    parser.add_argument("--data_file", default="../data/thai_edu_qa.jsonl")
    parser.add_argument("--output_dir", default="./checkpoints/typhoon-thai-edu")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=128,
                        help="LoRA rank — matches the reference paper")
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_file}")
    rows = load_jsonl(Path(args.data_file))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows   = [r for r in rows if r["split"] == "val"]
    print(f"  train: {len(train_rows)}, val: {len(val_rows)}")

    train_ds = Dataset.from_list(train_rows).map(format_example)
    val_ds   = Dataset.from_list(val_rows).map(format_example)

    print(f"\nLoading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA -----------------------------------------------------------
    print("\nApplying LoRA...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Tokenize -------------------------------------------------------------
    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    # Training -------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final LoRA adapter ---------------------------------------------
    final_dir = Path(args.output_dir) / "final"
    print(f"\nSaving final LoRA adapter to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    main()
