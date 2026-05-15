"""Re-evaluate trained ablation adapters with fixed generation params (no repetition_penalty, ngram=6)."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.run_ablation_matrix import (
    AblationConfig, run_quality_eval,
    load_jsonl, split_indices, subset_records, maybe_limit,
)

OUTPUT_DIR = Path("output/ablations_qwen35_0.8b")
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
config = AblationConfig(model_name=MODEL_NAME)

print("Loading holdout data...")
base_records = load_jsonl(Path("data/train_clean_for_model_967.jsonl"))
_, holdout_indices = split_indices(len(base_records), config.holdout_fraction, config.seed)
holdout_base = maybe_limit(subset_records(base_records, holdout_indices), config.max_holdout_records)
print(f"Holdout records: {len(holdout_base)}")

summary = {}
for exp_dir in sorted(OUTPUT_DIR.iterdir()):
    adapter_path = exp_dir / "adapter"
    if not adapter_path.exists():
        continue
    name = exp_dir.name
    print(f"\n--- {name} ---")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    report = run_quality_eval(model, tokenizer, holdout_base, config)
    summary[name] = report["avg_quality"]
    print(f"  avg_quality: {report['avg_quality']}")
    print(f"  sample[0] generated:\n    {report['samples'][0]['generated'][:300]}")

    del model, base_model
    torch.cuda.empty_cache()

print("\n=== RESULTS ===")
for name, q in summary.items():
    print(f"  {name}: {q:.4f}")
