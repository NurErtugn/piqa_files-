import argparse
import json
import time
import urllib.request
from pathlib import Path

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["useful", "not_useful", "uncertain"],
        },
        "category": {
            "type": "string",
            "enum": [
                "gem",
                "hidden_gem",
                "hard_negative",
                "uncertain",
            ],
        },
        "physical_relevance": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
        },
        "transferability": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
        },
        "reliability": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
        },
        "useful_segments": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "risk_flags": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "medical_or_veterinary": {"type": "boolean"},
                "dangerous_exercise": {"type": "boolean"},
                "weapon_instruction": {"type": "boolean"},
                "unsupported_safety_claim": {"type": "boolean"},
                "software_only": {"type": "boolean"},
                "pure_marketing": {"type": "boolean"},
                "doubtful_reliability": {"type": "boolean"},
            },
            "required": [
                "medical_or_veterinary",
                "dangerous_exercise",
                "weapon_instruction",
                "unsupported_safety_claim",
                "software_only",
                "pure_marketing",
                "doubtful_reliability",
            ],
        },
    },
    "required": [
        "decision",
        "category",
        "physical_relevance",
        "transferability",
        "reliability",
        "useful_segments",
        "risk_flags",
    ],
}

PROMPT = """You are performing a blinded independent audit of web documents
for training a model on physical-world commonsense.

You have not been given any previous classifier scores or decisions.
Judge this document independently.

A useful document must contain at least one substantial passage that teaches
reliable and transferable physical-world knowledge. Examples include physical
procedures, cooking transformations, repair, assembly, gardening, maintenance,
material behavior, object operation, troubleshooting, experimental findings,
or clearly explained physical mechanisms.

Definitions:

gem:
Most substantive content is reliable and reusable physical-world knowledge.

hidden_gem:
At least one substantial passage contains reliable, reusable physical-world
knowledge, but it is surrounded by advertising, narrative, repetition,
navigation, unrelated text, or other noise.

hard_negative:
No substantial reliable and transferable physical knowledge can be extracted,
or the potentially useful material is unsafe, unsupported, purely promotional,
software-only, or too unreliable.

uncertain:
Physical usefulness is plausible, but reliability or transferability cannot be
resolved confidently.

Rules:

- Do not accept a document merely because it mentions physical objects.
- Product pages qualify only when they teach generalizable mechanisms,
  constraints, material properties, maintenance, or troubleshooting.
- Recipes qualify when they contain meaningful procedures, transformations,
  adjustments, tests, or storage constraints.
- News qualifies only when it explains a reusable mechanism, engineering
  constraint, procedure, failure mode, or experimental result.
- Reject mere purchasing opinions, fashion preferences, event reports,
  isolated trivia, ingredient-only lists, and unsupported claims.
- Never select unsafe medical/veterinary advice, dangerous exercise
  instructions, weapon instructions, or unsupported safety-critical claims.
- useful_segments must contain only segment indices whose text independently
  supports the decision.
- A useful decision requires at least one substantial useful segment.
- Return uncertain rather than useful when evidence is ambiguous.
- Category and decision must follow these exact mappings:
  gem -> useful
  hidden_gem -> useful
  hard_negative -> not_useful
  uncertain -> uncertain
- A gem or hidden_gem classification requires at least one useful segment.
- Judge only the numbered segments below.
- Return only JSON matching the supplied schema.

SEGMENTS:
{segments}
"""


def request_judgment(endpoint, model, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1200,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "independent_physical_audit",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    }

    for attempt in range(5):
        attempt_payload = dict(payload)
        attempt_payload["seed"] = 1729 + attempt

        request = urllib.request.Request(
            endpoint,
            json.dumps(attempt_payload).encode(),
            {"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=300,
            ) as response:
                body = json.load(response)

            content = body["choices"][0]["message"]["content"]

            try:
                return json.loads(content)
            except json.JSONDecodeError as error:
                print(json.dumps({
                    "malformed_json_length": len(content),
                    "malformed_json_error": str(error),
                    "malformed_json_tail": content[-500:],
                }), flush=True)
                raise

        except Exception as error:
            print(
                f"Attempt {attempt + 1} failed: {error}",
                flush=True,
            )

            if attempt == 4:
                raise

            time.sleep(min(30, 2 ** attempt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_name(
        output_path.stem + ".errors.jsonl"
    )

    completed = set()

    if args.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(
                        json.loads(line)["id"]
                    )

    written = 0

    with input_path.open(encoding="utf-8") as source:
        with output_path.open("a", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, 1):
                record = json.loads(line)

                if record["id"] in completed:
                    continue

                segments = record["segments"]
                max_visible_segments = 24

                if len(segments) > max_visible_segments:
                    visible_indices = sorted({
                        round(
                            index * (len(segments) - 1)
                            / (max_visible_segments - 1)
                        )
                        for index in range(max_visible_segments)
                    })
                else:
                    visible_indices = list(range(len(segments)))

                rendered = "\n".join(
                    f"[{index}] {segments[index]}"
                    for index in visible_indices
                )

                try:
                    result = request_judgment(
                        args.endpoint,
                        args.model,
                        PROMPT.format(segments=rendered),
                    )
                except Exception as error:
                    error_record = {
                        "id": record["id"],
                        "line": line_number,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }

                    with error_path.open(
                        "a",
                        encoding="utf-8",
                    ) as error_destination:
                        error_destination.write(
                            json.dumps(
                                error_record,
                                ensure_ascii=False,
                            ) + "\n"
                        )

                    print(json.dumps({
                        "line": line_number,
                        "id": record["id"],
                        "technical_failure": str(error),
                    }), flush=True)
                    continue

                valid_indices = sorted({
                    index
                    for index in result["useful_segments"]
                    if isinstance(index, int)
                    and 0 <= index < len(record["segments"])
                })

                result["useful_segments"] = valid_indices

                category_to_decision = {
                    "gem": "useful",
                    "hidden_gem": "useful",
                    "hard_negative": "not_useful",
                    "uncertain": "uncertain",
                }

                result["decision"] = category_to_decision[
                    result["category"]
                ]

                if (
                    result["decision"] == "useful"
                    and not valid_indices
                ):
                    result["decision"] = "uncertain"
                    result["category"] = "uncertain"
                    result["reason"] = (
                        "Positive category lacked grounded segments."
                    )

                output_record = dict(record)
                output_record["independent_judge"] = result
                output_record["independent_judge_sampling"] = {
                    "total_segments": len(segments),
                    "visible_segments": len(visible_indices),
                    "visible_indices": visible_indices,
                    "sampling_applied": (
                        len(visible_indices) < len(segments)
                    ),
                }

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
                    "id": record["id"],
                    "decision": result["decision"],
                    "written": written,
                }), flush=True)

    print(json.dumps({
        "new_records_written": written,
        "output": str(output_path),
    }))


if __name__ == "__main__":
    main()
