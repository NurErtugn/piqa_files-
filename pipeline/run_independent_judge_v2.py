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
        "reason": {"type": "string"},
    },
    "required": [
        "decision",
        "category",
        "physical_relevance",
        "transferability",
        "reliability",
        "useful_segments",
        "risk_flags",
        "reason",
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
        "temperature": 0,
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

    request = urllib.request.Request(
        endpoint,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )

    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                request,
                timeout=300,
            ) as response:
                body = json.load(response)

            content = body["choices"][0]["message"]["content"]
            return json.loads(content)

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

    completed = set()

    if args.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(
                        json.loads(line)["blinded_id"]
                    )

    written = 0

    with input_path.open(encoding="utf-8") as source:
        with output_path.open("a", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, 1):
                record = json.loads(line)

                if record["blinded_id"] in completed:
                    continue

                rendered = "\n".join(
                    f"[{index}] {segment}"
                    for index, segment in enumerate(
                        record["segments"]
                    )
                )

                result = request_judgment(
                    args.endpoint,
                    args.model,
                    PROMPT.format(segments=rendered),
                )

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
                        "Positive category lacked grounded segments. "
                        + result["reason"]
                    )

                output_record = dict(record)
                output_record["independent_judge"] = result

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
                    "blinded_id": record["blinded_id"],
                    "decision": result["decision"],
                    "written": written,
                }), flush=True)

    print(json.dumps({
        "new_records_written": written,
        "output": str(output_path),
    }))


if __name__ == "__main__":
    main()
