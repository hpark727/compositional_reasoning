"""
Convert a raw curriculum JSON file into the Hugging Face dataset format
expected by the KG-guided SFT/RL pipeline.

The converter preserves the original KG fields, adds:
- text: Qwen-style chat text consumed by GRPO preprocessing
- answer: extracted A-D answer
- category: configurable metadata used by diversity filtering

Example:
    python3 convert_curriculum_dataset.py \
        --input_path curriculum_dataset_final.json \
        --output_path datasets/network_curriculum \
        --category_mode hop
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_PATH_KEYS = {"start", "relation", "end"}


def extract_tag(text: str, tag: str, required: bool = True) -> str:
    """Extract content from a simple XML-like tag."""
    start = f"<{tag}>"
    end = f"</{tag}>"
    if start not in text or end not in text:
        if required:
            raise ValueError(f"missing {start} or {end}")
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def extract_answer_block(text: str) -> str:
    """Extract the answer block, accepting <Answer>: and <Answer> forms."""
    if "<Answer>:" in text:
        block = text.split("<Answer>:", 1)[1].split("</Answer>", 1)[0]
    elif "<Answer>" in text:
        block = text.split("<Answer>", 1)[1].split("</Answer>", 1)[0]
    else:
        raise ValueError("missing <Answer> tag")

    letters = re.findall(r"\b[A-D]\b", block, flags=re.IGNORECASE)
    if not letters:
        raise ValueError("answer block does not contain A-D")
    return letters[-1].upper()


def parse_question_and_explanation(qae: str) -> Tuple[str, str, str, str]:
    """Parse the dataset's tagged question/options/explanation/answer field."""
    question = extract_tag(qae, "Question")
    options = extract_tag(qae, "Options")
    explanation = extract_tag(qae, "Explanation")
    answer = extract_answer_block(qae)
    return question, options, explanation, answer


def validate_paths(paths: Any) -> None:
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list")
    for path in paths:
        if not isinstance(path, dict):
            raise ValueError("each path entry must be a dict")
        missing = REQUIRED_PATH_KEYS - set(path)
        if missing:
            raise ValueError(f"path entry missing keys: {sorted(missing)}")
        for key in REQUIRED_PATH_KEYS:
            if path[key] in (None, ""):
                raise ValueError(f"path entry has empty {key!r}")


def build_text(question: str, options: str, explanation: str, answer: str) -> str:
    """Create the Qwen chat-style text consumed by preprocess_grpo_dataset."""
    user_content = f"{question.strip()}\nOptions:\n{options.strip()}"
    assistant_content = f"<think>\n{explanation.strip()}\n</think>\nFinal Answer: {answer}"
    return (
        "<|im_start|>user\n"
        f"{user_content}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_content}"
        "<|im_end|>"
    )


def build_category(record: Dict[str, Any], mode: str, constant: str) -> str:
    """Create the category field used by create_filtered_dataset.py."""
    if mode == "existing":
        if "category" not in record:
            raise ValueError("category_mode=existing requires a category field")
        return str(record["category"])

    paths = record.get("paths") or []
    if mode == "hop":
        return f"{record.get('k_hops', len(paths))}_hop"
    if mode == "first_relation":
        return str(paths[0]["relation"]) if paths else constant
    if mode == "relation_pattern":
        return " -> ".join(str(path["relation"]) for path in paths)
    if mode == "constant":
        return constant

    raise ValueError(f"unknown category mode: {mode}")


def convert_record(record: Dict[str, Any], category_mode: str, constant_category: str) -> Dict[str, Any]:
    required = ["id", "source_concept", "target_concept", "paths", "question_and_explanation"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"missing top-level fields: {missing}")

    validate_paths(record["paths"])
    question, options, explanation, answer = parse_question_and_explanation(
        record["question_and_explanation"]
    )

    converted = dict(record)
    converted["answer"] = answer
    converted["category"] = build_category(record, category_mode, constant_category)
    converted["text"] = build_text(question, options, explanation, answer)
    return converted


def load_json_records(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input JSON must be a list of records")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError("all input records must be JSON objects")
    return data


def convert_records(
    records: List[Dict[str, Any]],
    category_mode: str,
    constant_category: str,
    drop_invalid: bool,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]]]:
    converted: List[Dict[str, Any]] = []
    errors: List[Tuple[int, str]] = []

    for idx, record in enumerate(records):
        try:
            converted.append(convert_record(record, category_mode, constant_category))
        except Exception as exc:
            errors.append((idx, str(exc)))
            if not drop_invalid:
                break

    return converted, errors


def print_summary(converted: List[Dict[str, Any]], errors: List[Tuple[int, str]]) -> None:
    from collections import Counter

    print(f"Converted records: {len(converted)}")
    print(f"Invalid records: {len(errors)}")
    if errors:
        print("First invalid records:")
        for idx, error in errors[:5]:
            print(f"  {idx}: {error}")

    if converted:
        print("Columns:", ", ".join(sorted(converted[0].keys())))
        print("Answer distribution:", dict(Counter(record["answer"] for record in converted)))
        print("Category distribution:", dict(Counter(record["category"] for record in converted).most_common(10)))
        print("Hop distribution:", dict(Counter(record.get("k_hops") for record in converted)))


def save_hf_dataset(
    converted: List[Dict[str, Any]],
    output_path: Path,
    test_ratio: float,
    seed: int,
    overwrite: bool,
) -> None:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install repo dependencies first, for example:\n"
            "  python3 -m pip install -r requirements.txt\n"
            "Then rerun this converter."
        ) from exc

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)

    dataset = Dataset.from_list(converted)
    if test_ratio > 0:
        split = dataset.train_test_split(test_size=test_ratio, seed=seed)
        dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})
    else:
        dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(str(output_path))
    print(f"Saved Hugging Face dataset to {output_path}")
    print(dataset_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert curriculum JSON into this repo's Hugging Face dataset format."
    )
    parser.add_argument(
        "--input_path",
        default="curriculum_dataset_final.json",
        help="Path to the raw curriculum JSON list.",
    )
    parser.add_argument(
        "--output_path",
        default="datasets/network_curriculum",
        help="Directory where the Hugging Face dataset should be saved.",
    )
    parser.add_argument(
        "--category_mode",
        choices=["hop", "first_relation", "relation_pattern", "constant", "existing"],
        default="hop",
        help="How to synthesize the category column required by filtering.",
    )
    parser.add_argument(
        "--constant_category",
        default="computer_networks",
        help="Category value used when --category_mode constant is selected.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Optional held-out test ratio. Default keeps every record in train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for optional train/test split.",
    )
    parser.add_argument(
        "--drop_invalid",
        action="store_true",
        help="Drop invalid records instead of stopping at the first validation error.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and summarize without writing a dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test_ratio must be in [0, 1)")

    records = load_json_records(input_path)
    converted, errors = convert_records(
        records=records,
        category_mode=args.category_mode,
        constant_category=args.constant_category,
        drop_invalid=args.drop_invalid,
    )

    print_summary(converted, errors)

    if errors and not args.drop_invalid:
        raise SystemExit("Conversion stopped because invalid records were found.")
    if args.dry_run:
        return
    if not converted:
        raise SystemExit("No valid records to save.")

    save_hf_dataset(
        converted=converted,
        output_path=output_path,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
