"""
evaluate.py — Final Test-Set Evaluation for Fine-Tuned LoRA Model

Loads a trained LoRA adapter, runs inference on the test set, and computes
Exact Match (EM) and token-level F1. Saves predictions for error analysis.

USAGE
-----
    python evaluate.py \\
        --adapter_dir ./checkpoints/typhoon-thai-edu-lora/final \\
        --data_dir ./data \\
        --output_dir ./eval_results

REQUIREMENTS
------------
    pip install transformers peft accelerate pythainlp pandas

This script REQUIRES a GPU (the script that produced the adapter ran on GPU,
and inference on a 7B model is impractical on CPU).

Notes
-----
- Uses the SAME PyThaiNLP newmm tokenizer for F1 as in training preprocessing,
  per the framework's tokenizer-consistency requirement.
- Generates one prediction per test example with greedy decoding by default;
  use --temperature > 0 and --num_samples > 1 for multi-sample evaluation.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from pythainlp.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default base model — must match the one used during fine-tuning
DEFAULT_BASE_MODEL = "scb10x/typhoon-7b"

# Same prompt template as used in finetune_lora.py
PROMPT_TEMPLATE = """### คำสั่ง (Instruction):
ตอบคำถามต่อไปนี้โดยอ้างอิงจากบริบทที่ให้

### บริบท (Context):
{context}

### คำถาม (Question):
{question}

### คำตอบ (Answer):
"""


# ============================================================================
# Text-cleaning & metrics — must match what we use during training
# ============================================================================

def clean_thai(text: str) -> str:
    """Match the preprocessing pipeline exactly (notebook section B.1)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_answer(s: str) -> str:
    """Lowercase, strip whitespace, remove common punctuation."""
    s = clean_thai(s)
    s = re.sub(r"[\.\,\!\?\:\;\(\)\[\]\"\\\'\u2018\u2019\u201c\u201d]", "", s)
    return s.strip().lower()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score_tokens(pred: str, gold: str) -> float:
    """Token-level F1 using PyThaiNLP newmm — MUST match training tokenizer."""
    pred_toks = [t for t in word_tokenize(normalize_answer(pred), engine="newmm") if t.strip()]
    gold_toks = [t for t in word_tokenize(normalize_answer(gold), engine="newmm") if t.strip()]

    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)

    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# Sanity-test the metrics before trusting them on real predictions
# ============================================================================

def run_metric_sanity_tests():
    """Same tests as notebook section C.1 — run them every time."""
    assert exact_match("สวัสดี", "สวัสดี") == 1.0
    assert exact_match("สวัสดี", "สวัสดีครับ") == 0.0
    assert f1_score_tokens("สวัสดี", "สวัสดี") == 1.0
    assert 0.0 < f1_score_tokens("GPA ต้องไม่ต่ำกว่า 2.00",
                                  "นิสิตต้องมี GPA ไม่ต่ำกว่า 2.00") < 1.0
    print("  ✓ Metric sanity tests passed.")


# ============================================================================
# Model loading & inference
# ============================================================================

def load_model_and_tokenizer(base_model: str, adapter_dir: str):
    """Load base model + LoRA adapter, return model + tokenizer."""
    print(f"\n[1/3] Loading tokenizer from {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/3] Loading base model {base_model} (bfloat16)")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"[3/3] Merging LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict_one(model, tokenizer, context: str, question: str,
                max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    """Generate one answer for a (context, question) pair."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the new tokens (after the prompt)
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    # The model sometimes continues into the next "### " block; cut it off
    for stop in ["###", "\n\n\n"]:
        if stop in decoded:
            decoded = decoded.split(stop)[0].strip()
    return decoded


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter_dir", required=True,
                    help="Path to the saved LoRA adapter (e.g., ./checkpoints/.../final)")
    ap.add_argument("--data_dir", default="./data",
                    help="Directory containing thai_edu_qa.jsonl")
    ap.add_argument("--output_dir", default="./eval_results")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: only evaluate the first N examples (for debugging)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Sanity-check the metrics first -----
    print("Running metric sanity tests...")
    run_metric_sanity_tests()

    # ----- Load the data -----
    data_path = Path(args.data_dir) / "thai_edu_qa.jsonl"
    print(f"\nLoading {data_path}")
    df = pd.read_json(data_path, lines=True)
    test_df = df[df["split"] == args.split].copy().reset_index(drop=True)
    if args.limit:
        test_df = test_df.head(args.limit)
    print(f"  Test examples: {len(test_df)}")

    # ----- Load the model -----
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_dir)

    # ----- Run inference -----
    print(f"\nRunning inference on {len(test_df)} examples...")
    predictions = []
    for i, row in test_df.iterrows():
        pred = predict_one(model, tokenizer,
                           context=row["context"],
                           question=row["question"],
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature)
        predictions.append(pred)
        if (i + 1) % 10 == 0 or i == len(test_df) - 1:
            print(f"  [{i+1}/{len(test_df)}] done")

    test_df["prediction"] = predictions

    # ----- Compute metrics -----
    em_scores = [exact_match(p, g) for p, g in zip(test_df["prediction"], test_df["answer"])]
    f1_scores = [f1_score_tokens(p, g) for p, g in zip(test_df["prediction"], test_df["answer"])]

    test_df["em"] = em_scores
    test_df["f1"] = f1_scores

    overall = {
        "split": args.split,
        "n_examples": len(test_df),
        "EM": round(sum(em_scores) / len(em_scores) * 100, 2),
        "F1": round(sum(f1_scores) / len(f1_scores) * 100, 2),
        "adapter": args.adapter_dir,
        "base_model": args.base_model,
        "temperature": args.temperature,
    }

    # ----- Save outputs -----
    pred_path = output_dir / f"predictions_{args.split}.jsonl"
    test_df[["id", "question", "answer", "prediction", "em", "f1"]].to_json(
        pred_path, orient="records", lines=True, force_ascii=False
    )

    metrics_path = output_dir / f"metrics_{args.split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    # ----- Report -----
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({args.split} set, n = {overall['n_examples']})")
    print("=" * 60)
    print(f"  Exact Match (EM): {overall['EM']:>6.2f}")
    print(f"  Token F1        : {overall['F1']:>6.2f}")
    print("=" * 60)
    print(f"\nPredictions saved to: {pred_path}")
    print(f"Metrics saved to:     {metrics_path}")


if __name__ == "__main__":
    main()
