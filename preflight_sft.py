"""
Preflight checks for launching SFT on Della.

This validates paths and lightweight dependencies without loading model weights.
Run it from the repository root before submitting the Slurm job.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def require_imports(modules: Iterable[str]) -> None:
    missing = []
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:
            missing.append(f"{module}: {exc}")
    if missing:
        fail("Missing/broken imports:\n  " + "\n  ".join(missing))
    ok("Python package imports resolved")


def check_cuda(require_cuda: bool) -> None:
    import torch

    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    print(f"torch={torch.__version__}, cuda_available={available}, cuda_device_count={count}")

    if require_cuda and not available:
        fail("CUDA is required but torch.cuda.is_available() is False")
    if require_cuda and count < 1:
        fail("CUDA is required but no GPUs are visible")

    if available:
        ok("CUDA is visible to PyTorch")
    else:
        warn("CUDA is not visible. This can be normal on a non-GPU login node.")


def check_repo_files(deepspeed_path: Path) -> None:
    for path in [Path("sft_training.py"), Path("data_prep.py"), deepspeed_path]:
        if not path.exists():
            fail(f"Required file does not exist: {path}")
    ok("Repository files are present")

    try:
        with deepspeed_path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception as exc:
        fail(f"DeepSpeed config is not valid JSON: {deepspeed_path}: {exc}")
    ok(f"DeepSpeed config parses: {deepspeed_path}")


def check_dataset(dataset_path: Path) -> None:
    if not dataset_path.exists():
        fail(f"Dataset path does not exist: {dataset_path}")

    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, DatasetDict):
        fail(f"Expected DatasetDict with train split, got {type(dataset).__name__}")
    if "train" not in dataset:
        fail(f"DatasetDict has no train split. Splits: {list(dataset.keys())}")

    train = dataset["train"]
    columns = set(train.column_names)
    required = {
        "question_and_explanation",
        "text",
        "paths",
        "answer",
        "category",
        "source_concept",
        "target_concept",
    }
    missing = sorted(required - columns)
    if missing:
        fail(f"Dataset train split is missing columns: {missing}")
    if len(train) == 0:
        fail("Dataset train split is empty")

    row = train[0]
    if "<|im_start|>user\n" not in row["text"] or "<|im_start|>assistant\n" not in row["text"]:
        fail("Dataset text column does not look like Qwen chat text")
    if row["answer"] not in {"A", "B", "C", "D"}:
        fail(f"First row answer is not A-D: {row['answer']!r}")
    if not isinstance(row["paths"], list) or not row["paths"]:
        fail("First row paths is not a non-empty list")

    ok(f"Dataset loads: {dataset_path} ({len(train)} train rows)")
    print(f"Dataset columns: {train.column_names}")


def check_model(model_name: str, cache_dir: str | None, local_files_only: bool) -> None:
    from transformers import AutoConfig, AutoTokenizer

    kwargs = {"trust_remote_code": True, "local_files_only": local_files_only}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
    except Exception as exc:
        mode = "local cache" if local_files_only else "Hugging Face/local cache"
        fail(f"Could not resolve model/tokenizer from {mode}: {model_name}: {exc}")

    ok(f"Model config resolves: {model_name} ({config.model_type})")
    ok(f"Tokenizer resolves: vocab_size={getattr(tokenizer, 'vocab_size', 'unknown')}")


def check_output_dir(output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        warn(f"Output directory already exists and is non-empty: {output_dir}")
    else:
        ok(f"Output directory is available: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for SFT training.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B")
    parser.add_argument("--dataset_path", default="datasets/network_curriculum")
    parser.add_argument("--output_dir", default="sft_models/qwen3-14b-network-lora")
    parser.add_argument("--deepspeed", default="configs/deepspeed_config.json")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require model/tokenizer to already be in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Fail if PyTorch cannot see CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"cwd={Path.cwd()}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', '')}")
    print(f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE', '')}")
    print(f"CUDA_HOME={os.environ.get('CUDA_HOME', '')}")

    require_imports(["torch", "transformers", "datasets", "trl", "peft", "deepspeed"])
    check_cuda(require_cuda=args.require_cuda)
    check_repo_files(Path(args.deepspeed))
    check_dataset(Path(args.dataset_path))
    check_model(args.model_name, args.cache_dir, args.local_files_only)
    check_output_dir(Path(args.output_dir))

    ok("Preflight checks passed")


if __name__ == "__main__":
    main()
