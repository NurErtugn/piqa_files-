import argparse
import json
import time
import urllib.request
from pathlib import Path


SCHEMA = {
    "type": "object",
    "properties": {
        "final_training_text": {
            "type": "string",
            "minLength": 1,
        }
    },
    "required": ["final_training_text"],
    "additionalProperties": False,
}


PROMPT = """Transform the grounded source passage into a clear,
beginner-friendly training passage about physical-world commonsense.

Perform these stages internally:

1. Faithfully identify the facts, actions, conditions, constraints,
   qualifications, comparisons, measurements, ordering, uncertainty, and
   safety information explicitly supported by the source.

2. Identify reusable physical-common-sense relationships already entailed by
   the source, such as:
   - action -> physical outcome
   - condition -> consequence
   - material or object property -> possible use
   - prerequisite -> next action
   - observation -> physical state
   - incorrect action -> failure
   - adjustment -> correction
   - physical cause -> observable effect

3. Express that supported knowledge as a coherent teaching passage that a
   beginner can understand.

Strict grounding rules:

- Use only information supported by the source.
- Do not use outside knowledge.
- Do not invent materials, tools, objects, measurements, steps, purposes,
  explanations, consequences, warnings, recommendations, or causes.
- Do not infer a missing step merely because it would normally be reasonable.
- Make implicit logic explicit only when it necessarily follows from the
  source.
- Preserve all important numbers, units, conditions, ordering, modality,
  uncertainty, exceptions, comparisons, and safety limitations.
- Do not turn possibility into certainty.
- Do not turn a description into advice unless the source provides advice.
- Do not make a narrow example sound universally applicable.
- Remove advertising, prices, navigation, calls to action, attribution,
  repetition, and irrelevant narrative.
- Do not create a multiple-choice question.
- Do not mention PIQA, the source, extraction, rewriting, or these rules.
- Do not output analysis, relation labels, notes, or commentary.
- Return only JSON matching the supplied schema.

If a beginner-friendly abstraction cannot be made without adding unsupported
information, provide a faithful, clearly organized version of the source.

GROUNDED SOURCE:
{source}
"""


def request_transformation(
    endpoint,
    model,
    grounded_source,
    retries=5,
):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(
                    source=grounded_source
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_commonsense_rewrite",
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

            final_text = result["final_training_text"].strip()

            if not final_text:
                raise ValueError("Empty final_training_text")

            return final_text

        except Exception as error:
            last_error = error

            print(
                f"Attempt {attempt + 1} failed: {error}",
                flush=True,
            )

            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(str(last_error))


def load_completed_ids(output_path, error_path):
    completed = set()

    for path in (output_path, error_path):
        if not path.exists():
            continue

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(json.loads(line)["id"])

    return completed


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

    completed = (
        load_completed_ids(output_path, error_path)
        if args.resume
        else set()
    )

    written = 0
    failed = 0
    skipped = 0

    with input_path.open(encoding="utf-8") as source:
        with output_path.open("a", encoding="utf-8") as output:
            with error_path.open("a", encoding="utf-8") as errors:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    record_id = record["id"]

                    if record_id in completed:
                        skipped += 1
                        continue

                    grounded = record["training_text"].strip()

                    if not grounded:
                        error = "Empty grounded training_text"

                        errors.write(json.dumps({
                            "id": record_id,
                            "line": line_number,
                            "error": error,
                        }) + "\n")
                        errors.flush()

                        failed += 1
                        continue

                    try:
                        transformed = request_transformation(
                            args.endpoint,
                            args.model,
                            grounded,
                        )

                        output_record = dict(record)

                        # Preserve immutable evidence and all representations.
                        output_record["grounded_extraction"] = grounded
                        output_record[
                            "commonsense_training_candidate"
                        ] = transformed
                        output_record[
                            "commonsense_rewrite_model"
                        ] = args.model
                        output_record[
                            "commonsense_rewrite_status"
                        ] = "candidate_pending_validation"

                        output.write(
                            json.dumps(
                                output_record,
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        output.flush()

                        written += 1

                        print(json.dumps({
                            "line": line_number,
                            "id": record_id,
                            "written": written,
                        }), flush=True)

                    except Exception as error:
                        errors.write(
                            json.dumps({
                                "id": record_id,
                                "line": line_number,
                                "error": str(error),
                            }, ensure_ascii=False) + "\n"
                        )
                        errors.flush()

                        failed += 1

                        print(json.dumps({
                            "line": line_number,
                            "id": record_id,
                            "technical_failure": str(error),
                        }), flush=True)

    print(json.dumps({
        "written": written,
        "failed": failed,
        "skipped": skipped,
        "output": str(output_path),
        "errors": str(error_path),
    }), flush=True)


if __name__ == "__main__":
    main()
