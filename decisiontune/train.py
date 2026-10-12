"""LoRA supervised fine-tuning for decision models.

Approach (the standard community recipe, cf. the JevBench "Gemma 4 E2B LoRA"
entry): supervised fine-tuning on (rendered state+question -> correct option)
pairs with a PEFT LoRA adapter on a causal-LM backbone. Decision models like
Clef-flash are Qwen-family decoders, so this applies directly; the adapter is
served merged or via peft by `decisiontune serve`.

Heavy imports (torch/transformers/peft/datasets) are lazy: `plan()` and
`--dry-run` work anywhere with stdlib only; `train()` raises a clear error
telling you what to pip install.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dataset import Example
from .prompts import to_training_rows


@dataclass
class TrainConfig:
    model_id: str = "Clef-Flash-9B"  # HF id, e.g. the Cloudflare Clef-flash repo
    out_dir: str = "adapters/run1"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 2048
    seed: int = 42
    load_in_4bit: bool = True  # QLoRA: fits a 9B model on a 24GB GPU


def build_rows(examples: List[Example]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ex in examples:
        rows.extend(to_training_rows(ex.state, ex.questions, ex.answers))
    return rows


def plan(examples: List[Example], cfg: TrainConfig) -> Dict[str, Any]:
    """Describe what training would do, without touching torch."""
    rows = build_rows(examples)
    by_type: Dict[str, int] = {}
    for r in rows:
        by_type[r["question_type"]] = by_type.get(r["question_type"], 0) + 1
    longest = max((len(r["prompt"]) + len(r["completion"]) for r in rows), default=0)
    eff_batch = cfg.batch_size * cfg.grad_accum
    steps_per_epoch = max(1, (len(rows) + eff_batch - 1) // eff_batch)
    return {
        "model_id": cfg.model_id,
        "out_dir": cfg.out_dir,
        "train_rows": len(rows),
        "by_question_type": by_type,
        "longest_row_chars": longest,
        "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout,
                 "targets": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"},
        "optimizer": {"lr": cfg.learning_rate, "epochs": cfg.epochs,
                      "batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum,
                      "effective_batch": eff_batch},
        "est_steps": steps_per_epoch * cfg.epochs,
        "quantization": "4-bit QLoRA" if cfg.load_in_4bit else "bf16",
        "sample_prompt": rows[0]["prompt"][:600] if rows else "",
        "sample_completion": rows[0]["completion"] if rows else "",
    }


def train(examples: List[Example], cfg: TrainConfig) -> str:
    """Run real LoRA training. Requires torch, transformers, peft, datasets."""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  TrainingArguments, Trainer,
                                  DataCollatorForLanguageModeling)
    except ImportError as e:
        raise RuntimeError(
            "Training needs: pip install torch transformers peft datasets bitsandbytes\n"
            "(see README 'Real-model training' for the exact commands)"
        ) from e

    rows = build_rows(examples)
    if not rows:
        raise ValueError("no training rows built from examples")

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True,
                                    "torch_dtype": torch.bfloat16,
                                    "device_map": "auto"}
    if cfg.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    def tokenize(row):
        text = row["prompt"] + row["completion"] + tok.eos_token
        enc = tok(text, truncation=True, max_length=cfg.max_seq_len)
        # mask the prompt: only the completion contributes to the loss
        prompt_ids = tok(row["prompt"], truncation=True,
                         max_length=cfg.max_seq_len)["input_ids"]
        labels = [-100] * len(prompt_ids) + enc["input_ids"][len(prompt_ids):]
        enc["labels"] = labels[:len(enc["input_ids"])]
        return enc

    ds = Dataset.from_list(rows).map(tokenize, remove_columns=[c for c in rows[0]])
    args = TrainingArguments(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        seed=cfg.seed,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    return cfg.out_dir
