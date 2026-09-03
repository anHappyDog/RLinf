"""Run B1K VQA generation or independent judging through an SGLang API."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import load_config, make_choices, parse_json_object, read_jsonl

LOGGER = logging.getLogger(__name__)
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _request_payload(
    record: dict[str, Any],
    mode: str,
    system_prompt: str,
    server_config: dict[str, Any],
) -> dict[str, Any]:
    if mode == "generate":
        task = {
            "question_type": record["question_type"],
            "canonical_question": record["question"],
            "correct_label": record["correct_label"],
            "allowed_distractors": record["distractor_pool"],
            "required_distractor_count": int(server_config["choice_count"]) - 1,
            "task_goal": record["goal"]
            if record["question_type"] == "role_identification"
            else None,
            "current_primitive": record["skill"]
            if record["question_type"] == "role_identification"
            else None,
        }
    else:
        task = {
            "question_type": record["question_type"],
            "question": record["question"],
            "choices": [
                f"{chr(65 + index)}. {choice}"
                for index, choice in enumerate(record["choices"])
            ],
        }

    payload: dict[str, Any] = {
        "model": server_config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(Path(record["image_path"]))
                        },
                    },
                    {
                        "type": "text",
                        "text": json.dumps(task, ensure_ascii=False),
                    },
                ],
            },
        ],
        "temperature": float(server_config["temperature"]),
        "max_tokens": int(server_config["max_tokens"]),
    }
    if server_config.get("json_response_format", True):
        payload["response_format"] = {"type": "json_object"}
    if server_config.get("disable_thinking", True):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def _call_api(payload: dict[str, Any], server_config: dict[str, Any]) -> str:
    endpoint = f"{server_config['base_url'].rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key_env = server_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    proxy_handler = (
        urllib.request.ProxyHandler()
        if server_config.get("use_environment_proxy", False)
        else urllib.request.ProxyHandler({})
    )
    opener = urllib.request.build_opener(proxy_handler)
    retries = int(server_config["retries"])
    for attempt in range(retries + 1):
        try:
            with opener.open(
                request, timeout=float(server_config["timeout_seconds"])
            ) as response:
                body = json.load(response)
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_HTTP_CODES or attempt == retries:
                detail = error.read().decode(errors="replace")
                raise RuntimeError(f"SGLang HTTP {error.code}: {detail}") from error
        except urllib.error.URLError:
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    raise AssertionError("Retry loop exited unexpectedly")


def _validate_generator(
    record: dict[str, Any],
    response: dict[str, Any],
    choice_count: int,
    seed: int,
    restrict_distractors_to_pool: bool,
) -> dict[str, Any]:
    question = response.get("question")
    distractors = response.get("distractors")
    visually_answerable = response.get("visually_answerable")
    ambiguous = response.get("ambiguous")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generator returned an empty question")
    leaked_label = re.search(
        rf"(?<!\w){re.escape(record['correct_label'])}(?!\w)",
        question,
        flags=re.IGNORECASE,
    )
    if leaked_label:
        raise ValueError("Generated question leaks the correct label")
    if not isinstance(distractors, list) or not all(
        isinstance(value, str) for value in distractors
    ):
        raise ValueError("Generator distractors must be a string list")
    if len(distractors) != choice_count - 1:
        raise ValueError(
            f"Generator returned {len(distractors)} distractors; "
            f"expected {choice_count - 1}"
        )
    if not isinstance(visually_answerable, bool) or not isinstance(ambiguous, bool):
        raise ValueError("Generator quality flags must be booleans")

    if restrict_distractors_to_pool:
        allowed_by_key = {
            value.casefold(): value for value in record["distractor_pool"]
        }
        canonical_distractors = []
        for distractor in distractors:
            key = distractor.strip().casefold()
            if key not in allowed_by_key:
                raise ValueError(
                    f"Distractor is outside the allowed pool: {distractor!r}"
                )
            canonical_distractors.append(allowed_by_key[key])
    else:
        canonical_distractors = [distractor.strip() for distractor in distractors]
    choices, correct_answer = make_choices(
        record["id"],
        record["correct_label"],
        canonical_distractors,
        choice_count,
        seed,
    )
    output = dict(record)
    output.update(
        {
            "question": question.strip(),
            "choices": choices,
            "correct_answer": correct_answer,
            "generator": {
                "visually_answerable": visually_answerable,
                "ambiguous": ambiguous,
                "reason": str(response.get("reason", "")),
                "distractors": canonical_distractors,
            },
        }
    )
    return output


def _validate_judge(record: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    predicted_answer = str(response.get("predicted_answer", "")).strip().upper()
    visually_answerable = response.get("visually_answerable")
    ambiguous = response.get("ambiguous")
    valid_letters = "ABCDE"[: len(record["choices"])]
    if predicted_answer not in valid_letters:
        raise ValueError(f"Judge returned invalid answer {predicted_answer!r}")
    if not isinstance(visually_answerable, bool) or not isinstance(ambiguous, bool):
        raise ValueError("Judge quality flags must be booleans")
    output = dict(record)
    output["judge"] = {
        "predicted_answer": predicted_answer,
        "visually_answerable": visually_answerable,
        "ambiguous": ambiguous,
        "reason": str(response.get("reason", "")),
    }
    return output


def _process_record(
    record: dict[str, Any],
    mode: str,
    system_prompt: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _request_payload(record, mode, system_prompt, config["sglang"])
    response_text = _call_api(payload, config["sglang"])
    response = parse_json_object(response_text)
    if mode == "generate":
        return _validate_generator(
            record,
            response,
            int(config["sglang"]["choice_count"]),
            int(config["seed"]),
            bool(config["sglang"]["restrict_distractors_to_pool"]),
        )
    return _validate_judge(record, response)


def run(
    config: dict[str, Any],
    mode: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> tuple[int, int]:
    """Process an input JSONL with resume support."""
    if overwrite and output_path.exists():
        output_path.unlink()
    completed_ids = (
        {record["id"] for record in read_jsonl(output_path)}
        if output_path.exists()
        else set()
    )
    records = [
        record for record in read_jsonl(input_path) if record["id"] not in completed_ids
    ]
    prompt_path = Path(config["prompts"][mode])
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    succeeded = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(
            max_workers=int(config["sglang"]["concurrency"])
        ) as executor:
            future_to_record = {
                executor.submit(
                    _process_record, record, mode, system_prompt, config
                ): record
                for record in records
            }
            for completed, future in enumerate(as_completed(future_to_record), start=1):
                record = future_to_record[future]
                try:
                    output = future.result()
                except Exception as error:  # Preserve per-record failures for reruns.
                    errors.append(
                        {
                            "id": record["id"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    LOGGER.error("%s failed: %s", record["id"], error)
                    continue
                output_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                output_file.flush()
                succeeded += 1
                if completed % 100 == 0:
                    LOGGER.info("Processed %d/%d", completed, len(records))

    errors_path = output_path.with_suffix(".errors.jsonl")
    with errors_path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, ensure_ascii=False) + "\n")
    return succeeded, len(errors)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("generate", "judge"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected SGLang stage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    output_root = args.output_root or Path(config["output_root"])
    default_input = (
        output_root / "intermediate" / "candidates.jsonl"
        if args.mode == "generate"
        else output_root / "intermediate" / "generated.jsonl"
    )
    default_output = output_root / "intermediate" / f"{args.mode}d.jsonl"
    succeeded, failed = run(
        config,
        args.mode,
        args.input or default_input,
        args.output or default_output,
        args.overwrite,
    )
    LOGGER.info("SGLang %s: %d succeeded, %d failed", args.mode, succeeded, failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
