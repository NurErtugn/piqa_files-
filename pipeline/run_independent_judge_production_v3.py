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

Score the three dimensions independently, using the strongest numbered passage
that you select as useful. Do not copy one score into the other fields.

physical_relevance:
0 = no physical-world knowledge; 1 = physical objects are only mentioned or
the material is tangential; 2 = a substantial physical procedure, mechanism,
constraint, or transformation is explained; 3 = detailed, actionable physical
knowledge with concrete steps, conditions, or causal explanation.

transferability:
0 = no reusable lesson; 1 = an isolated fact, anecdote, product-specific claim,
or vague advice; 2 = a lesson usable for a similar object, material, or task
under similar conditions; 3 = a broadly reusable lesson with clear conditions
or explanations. A concrete recipe, repair procedure, or growing instruction
can score 2 when its steps and conditions are clear. The lesson does not need
to apply to every kind of cooking, repair, or gardening.

reliability:
0 = unsupported, contradictory, or clearly unreliable; 1 = evidence is weak,
ambiguous, or missing important qualifications; 2 = internally consistent and
grounded in the supplied passage; 3 = especially clear, specific, and well
supported by the supplied passage. Do not use outside knowledge to repair a
source claim.

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
- Assign the three scores before choosing the category. A gem or hidden_gem
  requires all three scores to be at least 2 and at least one substantial
  useful segment. Otherwise choose uncertain when a useful lesson is plausible
  but unresolved, or hard_negative when no qualifying lesson exists.
- A score of 1 in any dimension is not enough for a useful decision.
- Category and decision must follow these exact mappings:
  gem -> useful
  hidden_gem -> useful
  hard_negative -> not_useful
  uncertain -> uncertain
- A gem or hidden_gem classification requires at least one useful segment.
- Judge only the numbered segments below.
- Return only compact, single-line JSON matching the supplied schema.
- Do not pretty-print, indent, or add unnecessary whitespace.

SEGMENTS:
{segments}
"""


def request_judgment(endpoint, model, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2400,
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

                visible_set = set(visible_indices)
                raw_indices = result["useful_segments"]
                valid_indices = sorted({
                    index
                    for index in raw_indices
                    if isinstance(index, int)
                    and index in visible_set
                })

                result["useful_segments"] = valid_indices

                category_to_decision = {
                    "gem": "useful",
                    "hidden_gem": "useful",
                    "hard_negative": "not_useful",
                    "uncertain": "uncertain",
                }

                model_category = result["category"]
                model_decision = result["decision"]
                expected_decision = category_to_decision[model_category]
                scores = (
                    result["physical_relevance"],
                    result["transferability"],
                    result["reliability"],
                )
                positive = model_category in {"gem", "hidden_gem"}
                hard_risks = (
                    "medical_or_veterinary",
                    "dangerous_exercise",
                    "weapon_instruction",
                    "unsupported_safety_claim",
                )
                flags = []
                if model_decision != expected_decision:
                    flags.append("decision_category_mismatch")
                if len(valid_indices) != len(set(raw_indices)):
                    flags.append("invalid_or_invisible_useful_segment")
                if positive and not valid_indices:
                    flags.append("positive_without_grounded_segment")
                if positive and min(scores) < 2:
                    flags.append("positive_with_score_below_2")
                if positive and any(result["risk_flags"][key] for key in hard_risks):
                    flags.append("positive_with_hard_risk")
                if model_category == "hard_negative" and min(scores) >= 2 and valid_indices:
                    flags.append("negative_with_high_scores_and_segments")

                # Preserve the model's original outputs for audit. Contradictions
                # are quarantined as uncertain, never silently made useful.
                result["model_category"] = model_category
                result["model_decision"] = model_decision
                result["validation_flags"] = flags
                result["category"] = "uncertain" if flags else model_category
                result["decision"] = "uncertain" if flags else expected_decision

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
