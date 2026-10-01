import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--model",
        default="ellamind/propella-1-4b",
    )
    args = parser.parse_args()

    # Download only Propella's official prompt/schema code.
    code_directory = snapshot_download(
        repo_id=args.model,
        allow_patterns=[
            "propella.py",
            "property_descriptions.md",
        ],
    )

    original_directory = os.getcwd()
    os.chdir(code_directory)
    sys.path.insert(0, code_directory)

    from propella import (
        AnnotationResponse,
        create_messages,
        get_annotation_response_schema,
    )

    os.chdir(original_directory)

    schema = get_annotation_response_schema(
        flatten=True,
        compact_whitespace=False,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    errors_path = Path(args.errors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()

    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed_ids.add(json.loads(line)["id"])

    written = 0

    with input_path.open(encoding="utf-8") as source:
        with output_path.open("a", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, 1):
                record = json.loads(line)

                if record["id"] in completed_ids:
                    continue

                payload = {
                    "model": args.model,
                    "messages": create_messages(
                        record.get(
                            "propella_text_override",
                            record["text_clean"],
                        )[:12000]
                    ),
                    "temperature": 0,
                    "max_tokens": 2048,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "AnnotationResponse",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                }

                response_json = None

                for attempt in range(5):
                    try:
                        response = requests.post(
                            args.endpoint,
                            json=payload,
                            timeout=300,
                        )

                        if response.status_code != 200:
                            raise RuntimeError(
                                f"HTTP {response.status_code}: "
                                f"{response.text[:2000]}"
                            )

                        response_json = response.json()
                        content = response_json[
                            "choices"
                        ][0]["message"]["content"]

                        annotation = (
                            AnnotationResponse.model_validate_json(content)
                        )

                        result = {
                            **record,
                            "propella": annotation.model_dump(),
                        }

                        destination.write(
                            json.dumps(result, ensure_ascii=False) + "\n"
                        )
                        destination.flush()

                        written += 1
                        print(
                            json.dumps({
                                "line": line_number,
                                "id": record["id"],
                                "human_category": record.get("human_category"),
                                "written": written,
                            }),
                            flush=True,
                        )
                        break

                    except Exception as error:
                        print(
                            f"Attempt {attempt + 1} failed: {error}",
                            file=sys.stderr,
                            flush=True,
                        )

                        if attempt == 4:
                            error_record = {
                                "id": record["id"],
                                "line": line_number,
                                "error": str(error),
                            }

                            if response_json is not None:
                                error_record["response"] = response_json

                            with errors_path.open(
                                "a",
                                encoding="utf-8",
                            ) as error_file:
                                error_file.write(
                                    json.dumps(
                                        error_record,
                                        ensure_ascii=False,
                                    ) + "\n"
                                )
                                error_file.flush()

                            print(
                                json.dumps({
                                    "line": line_number,
                                    "id": record["id"],
                                    "technical_failure": str(error),
                                }),
                                flush=True,
                            )
                            break

                        time.sleep(2 ** attempt)

    print(
        json.dumps({
            "new_records_written": written,
            "output": str(output_path),
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
