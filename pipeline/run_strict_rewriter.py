import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_text": {
            "type": "string",
            "minLength": 1,
        }
    },
    "required": ["rewritten_text"],
    "additionalProperties": False,
}


PROMPT = """Rewrite the SOURCE into a clear, self-contained training passage
about physical-world commonsense.

The SOURCE has already been extracted from a noisy web document. Reorganize
and clarify it conservatively.

Strict requirements:

- Use only information explicitly supported by SOURCE.
- Do not add facts, explanations, causes, recommendations, warnings,
  materials, tools, measurements, numbers, units, steps, or conclusions.
- Do not infer missing steps.
- Preserve every important condition, qualification, uncertainty, exception,
  number, unit, comparison, and safety statement.
- Do not strengthen possibility into certainty or advice.
- Remove navigation, advertising, prices, calls to action, attribution,
  publication metadata, repetition, and incomplete fragments only when doing
  so does not remove useful physical knowledge.
- Prefer concise connected prose.
- Use numbered steps only when SOURCE explicitly describes an ordered
  procedure.
- Do not mention the source, document, extraction, rewriting, or these rules.
- If SOURCE is already clear, reproduce it with minimal editing.
- Return only JSON matching the supplied schema.

SOURCE:
{source}
"""


def request_rewrite(endpoint, model, source, retries=5):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(source=source),
            }
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "strict_physical_rewrite",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    }

    request = urllib.request.Request(
        endpoint,
        json.dumps(payload).encode("utf-8"),
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer ",
        },
    )

    last_error = None

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                request,
                timeout=300,
            ) as response:
                body = json.load(response)

            content = body["choices"][0]["message"]["content"]
            result = json.loads(content)

            rewritten = result["rewritten_text"].strip()

            if not rewritten:
                raise ValueError("Empty rewritten_text")

            return rewritten

        except Exception as error:
            last_error = error
            print(
                f"Attempt {attempt + 1} failed: {error}",
                flush=True,
            )

            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(str(last_error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = Path(args.errors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    completed = set()

    if args.resume:
        for path in (output_path, error_path):
            if not path.exists():
                continue

            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        completed.add(json.loads(line)["id"])

    written = 0
    failed = 0

    with input_path.open(encoding="utf-8") as source:
        with output_path.open("a", encoding="utf-8") as destination:
            with error_path.open("a", encoding="utf-8") as errors:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    record_id = record["id"]

                    if record_id in completed:
                        continue

                    grounded = record["training_text"].strip()

                    try:
                        rewritten = request_rewrite(
                            args.endpoint,
                            args.model,
                            grounded,
                        )

                        output_record = dict(record)
                        output_record["grounded_extraction"] = grounded
                        output_record["rewritten_text"] = rewritten
                        output_record["rewrite_model"] = args.model
                        output_record["rewrite_status"] = (
                            "candidate_pending_validation"
                        )

                        destination.write(
                            json.dumps(
                                output_record,
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        destination.flush()

                        written += 1

                        print(json.dumps({
                            "line": line_number,
                            "id": record_id,
                            "written": written,
                        }), flush=True)

                    except Exception as error:
                        error_record = {
                            "id": record_id,
                            "line": line_number,
                            "error": str(error),
                        }

                        errors.write(
                            json.dumps(
                                error_record,
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        errors.flush()

                        failed += 1

                        print(json.dumps({
                            "line": line_number,
                            "id": record_id,
                            "technical_failure": str(error),
                        }), flush=True)

    print(json.dumps({
        "new_records_written": written,
        "new_failures": failed,
        "output": str(output_path),
        "errors": str(error_path),
    }), flush=True)


if __name__ == "__main__":
    main()
