from __future__ import annotations

import json
import os
import time
import urllib.request

from .core import parse_bool, risk_flags, segment, token_count

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "task_relevance": {"type": "integer", "minimum": 0, "maximum": 3},
        "transferable_knowledge": {"type": "integer", "minimum": 0, "maximum": 3},
        "useful_segments": {"type": "array", "items": {"type": "integer"}},
        "source_type": {"type": "string", "enum": ["repair", "troubleshooting", "physical_procedure", "mechanism_explanation", "assembly", "tool_or_object_use", "comparison", "experiment", "manual", "product_constraints", "nonphysical"]},
        "risk_flags": {"type": "object", "additionalProperties": False, "properties": {k: {"type": "boolean"} for k in ["medical_or_veterinary", "dangerous_exercise", "weapon_instruction", "unsupported_safety_claim", "software_only", "pure_marketing", "doubtful_reliability"]}},
        "provisional_label": {
            "type": "string",
            "enum": [
                "gem_candidate",
                "hidden_gem_candidate",
                "hard_negative",
                "uncertain"
            ]
        },
        "decision": {
            "type": "string",
            "enum": ["accept", "uncertain", "reject"]
        }
    },
    "required": [
        "task_relevance",
        "transferable_knowledge",
        "useful_segments",
        "source_type",
        "risk_flags",
        "provisional_label",
        "decision"
    ]
}

PROMPT = """You judge web documents for reliable, transferable physical-world knowledge used in PIQA-style pretraining.

Return only JSON matching the supplied schema.

CORE DEFINITIONS

gem_candidate:
The document is predominantly reliable, useful physical-world knowledge. Its procedures, mechanisms, constraints, troubleshooting principles, or object-use information are broadly reusable. Most substantial segments are useful.

hidden_gem_candidate:
At least one substantial segment contains reliable, transferable physical knowledge, but useful information is surrounded by irrelevant material, personal narrative, advertising, navigation, repetition, or other semantic noise.

hard_negative:
No substantial safe and reliable transferable physical knowledge can be recovered, or the document is disallowed by a risk rule.

uncertain:
Potentially useful physical knowledge exists, but reliability, grounding, context, or interpretation is too ambiguous for a precision decision.

STRICT RELIABILITY RULES

- Reject quizzes, multiple-choice exercises, answer-key fragments, or pages containing contradictory candidate answers unless the document clearly explains and resolves the correct physical principle.
- Reject unsupported factual, medical, veterinary, exercise-safety, weapon-handling, or safety-critical claims.
- Never select medical or veterinary recommendations, weapon instructions, or dangerous exercise instructions.
- Their presence elsewhere does not automatically reject a document when separate, safe, non-medical physical knowledge can be cleanly isolated.
- Reject the whole document when the risky material is central, cannot be separated, or contaminates the candidate useful segments.
- Reject software-only material.
- Set pure_marketing true only when the page provides no independent, transferable physical knowledge beyond selling or praising something.
- Marketing surrounding otherwise useful material is semantic noise, not automatic whole-document rejection.
- A product-oriented document may qualify only when selected segments contain generalizable physical constraints, material properties, mechanisms, maintenance, comparison criteria, or troubleshooting information independent of the product pitch.
- News content qualifies only when selected segments clearly explain a transferable physical mechanism, procedure, experimental result, failure mechanism, or engineering constraint. Merely reporting that an event or experiment occurred is insufficient.
- Physical knowledge includes cooking, baking, material behavior, gardening, repair, maintenance, tool use, object operation, assembly, troubleshooting, and clearly explained physical mechanisms.
- Recipes qualify when they contain meaningful procedures, observable tests, adjustments, transformations, storage constraints, or cause-and-effect—not when they are merely ingredient lists.
- Reject narrow aesthetic preferences, fashion advice, purchasing opinions, and isolated trivia that lack a transferable physical principle.
- Personal experience is not evidence by itself. Select only independently understandable physical knowledge.
- Do not reward fluent writing when the content is ungrounded or contradictory.

SEGMENT SELECTION

- useful_segments must contain only indices of segments with independently understandable, reliable, transferable physical knowledge.
- Do not select navigation, advertisements, biography, unsupported claims, repetition, or mere product praise.
- Do not estimate a noise fraction. Noise is calculated deterministically afterward from selected segment tokens.
- An accept decision requires at least one substantial selected segment.
- If no segment is safe and reliable, return an empty useful_segments list.

MODE RULES

precision:
Accept only clear cases. Route plausible but unresolved cases to uncertain.

recall:
Recover a document when at least one substantial safe, reliable, transferable physical segment is present, even if the surrounding page is noisy. Risk and reliability rejection rules remain unchanged.

CONSISTENCY

- gem_candidate or hidden_gem_candidate -> decision accept
- hard_negative -> decision reject
- uncertain -> decision uncertain
- If a risk flag applies to selected candidate knowledge, remove those segments. If no substantial safe segment remains, return hard_negative/reject.
- Use uncertain—not accept—when physical relevance is plausible but reliability, transferability, or grounding cannot be confidently resolved.

Scores:
task_relevance:
0 no physical relevance
1 physical object merely mentioned
2 useful physical procedure, constraint, mechanism, or outcome
3 strong repair, tool use, assembly, troubleshooting, or cause-and-effect knowledge

transferable_knowledge:
0 none
1 isolated narrow fact
2 reusable procedure or constraint
3 general mechanism, diagnostic principle, or broadly reusable physical insight

MODE={mode}

SEGMENTS:
{segments}"""


class CompatibleJudge:
    """OpenAI-compatible chat-completions client; endpoint/model are configurable."""
    def __init__(self, endpoint: str | None = None, model: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint or os.environ.get("JUDGE_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.model = model or os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-32B")
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", "")

    def __call__(self, prompt: str, retries: int = 5) -> dict:
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                   "response_format": {"type": "json_schema", "json_schema": {"name": "physical_judge", "strict": True, "schema": SCHEMA}}}
        request = urllib.request.Request(self.endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.load(response)
                return json.loads(body["choices"][0]["message"]["content"])
            except Exception:
                if attempt + 1 == retries: raise
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError("unreachable")


def judge_record(record: dict, mode: str, client: CompatibleJudge) -> dict:
    parts = record.get("segments") or segment(record["text_clean"])

    max_visible_segments = 24

    if len(parts) > max_visible_segments:
        visible_indices = sorted({
            round(
                index * (len(parts) - 1)
                / (max_visible_segments - 1)
            )
            for index in range(max_visible_segments)
        })
    else:
        visible_indices = list(range(len(parts)))

    rendered = "\n".join(
        f"[{index}] {parts[index]}"
        for index in visible_indices
    )

    result = client(PROMPT.format(mode=mode, segments=rendered))
    indices = sorted({i for i in result.get("useful_segments", []) if isinstance(i, int) and 0 <= i < len(parts)})
    selected = [parts[i] for i in indices]
    total = sum(token_count(p) for p in parts); useful = sum(token_count(p) for p in selected)
    result["useful_segments"] = indices
    deterministic = risk_flags(record["text_clean"], result.get("source_type", ""))
    reported = {k: parse_bool(v) for k, v in result.get("risk_flags", {}).items()}
    result["risk_flags"] = {k: deterministic.get(k, False) or reported.get(k, False) for k in deterministic.keys() | reported.keys()}
    record = dict(record)
    record["segments"] = parts
    record[f"{mode}_judge"] = result
    record[f"{mode}_judge_sampling"] = {
        "total_segments": len(parts),
        "visible_segments": len(visible_indices),
        "visible_indices": visible_indices,
        "sampling_applied": len(visible_indices) < len(parts),
    }
    record["useful_tokens"] = useful; record["semantic_noise_fraction"] = 1.0 - useful / max(1, total)
    record["text_extracted"] = "\n".join(selected)
    return record
