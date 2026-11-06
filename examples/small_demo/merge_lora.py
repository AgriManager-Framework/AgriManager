"""Merge a LoRA adapter saved by the GRPO trainer into its base model.

Usage: python examples/small_demo/merge_lora.py --base <hf id or dir> \
           --adapter <.../actor/lora_adapter> --out <dir>

Writes a full Transformers checkpoint (bf16) plus the tokenizer to --out, which
vLLM can score directly. Runs on the CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    # lora_B starts at zero; if none of it is non-zero the adapter keys did not
    # load and merging would silently return the base model.
    lora_b = [p for name, p in model.named_parameters() if "lora_B" in name]
    trained = sum(int(p.detach().abs().sum().item() > 0) for p in lora_b)
    if not lora_b or trained == 0:
        raise SystemExit(f"no trained LoRA weights found in {args.adapter}")
    print(f"{trained} of {len(lora_b)} LoRA B matrices are non-zero")
    merged = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(args.out))
    print(f"merged {args.adapter} into {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
