from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t]+")
URL_ONLY_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.I)
BOILERPLATE_RE = re.compile(
    r"(?:cookie|privacy policy|accept all|sign in|log in|subscribe|newsletter|"
    r"all rights reserved|share (?:this|on)|follow us|advertisement|skip to content|"
    r"terms (?:of use|and conditions)|enable javascript)", re.I
)
CRAWL_RE = re.compile(r"(?:^WARC-|commoncrawl|crawl-data|urn:uuid:)", re.I)
PHYSICAL = {
    "attach", "bend", "bolt", "clamp", "clean", "cut", "drill", "fasten", "fit",
    "fold", "glue", "grip", "hammer", "heat", "insert", "joint", "lift", "loosen",
    "measure", "melt", "nail", "paint", "peel", "repair", "replace", "rotate", "sand",
    "saw", "screw", "seal", "sharpen", "slide", "tighten", "tool", "weld", "wire"
}
PROCEDURE = {"first", "next", "then", "after", "before", "until", "step", "finally"}
SOFTWARE = {"click", "browser", "app", "login", "password", "download", "install", "website"}
RISK_PATTERNS = {
    "medical_or_veterinary": re.compile(r"\b(?:diagnos|dosage|patient|veterinar|medication)\w*\b", re.I),
    "dangerous_exercise": re.compile(r"\b(?:extreme weight loss|train through pain|neck bridge)\b", re.I),
    "weapon_instruction": re.compile(r"\b(?:build|make|assemble) (?:a |an )?(?:bomb|weapon|silencer)\b", re.I),
    "unsupported_safety_claim": re.compile(r"\b(?:completely safe|guaranteed safe|zero risk)\b", re.I),
}


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def token_count(text: str) -> int:
    """Deterministic corpus token proxy; replace with training tokenizer if desired."""
    return len(WORD_RE.findall(text))


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_document(text: str) -> dict:
    raw = unicodedata.normalize("NFKC", text or "")
    raw = CONTROL_RE.sub("", raw)
    kept, removed, seen = [], [], set()
    for original in raw.splitlines():
        line = SPACE_RE.sub(" ", original).strip()
        normalized = line.casefold()
        reason = (
            not line or (len(words(line)) < 2 and len(line) < 12) or URL_ONLY_RE.match(line) or
            CRAWL_RE.search(line) or BOILERPLATE_RE.search(line) or normalized in seen
        )
        if reason:
            if line:
                removed.append(line)
            continue
        seen.add(normalized)
        kept.append(line)
    clean = "\n".join(kept)
    return {
        "text_raw": text or "",
        "text_clean": clean,
        "removed_text": "\n".join(removed),
        "mechanical_noise_fraction": 1.0 - token_count(clean) / max(1, token_count(text or "")),
    }


def segment(text: str, target_words: int = 100) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{1,}|(?<=[.!?])\s+(?=[A-Z0-9])", text) if p.strip()]
    result, current, size = [], [], 0
    for p in paragraphs:
        n = token_count(p)
        if current and size + n > target_words:
            result.append(" ".join(current)); current, size = [], 0
        current.append(p); size += n
    if current:
        result.append(" ".join(current))
    return result


def heuristic_features(text: str, removed_text: str = "", url: str = "") -> dict:
    ws = words(text); n = max(1, len(ws)); unique = len(set(ws)) / n
    physical = sum(w in PHYSICAL for w in ws) / n
    procedure = sum(w in PROCEDURE for w in ws) / n
    software = sum(w in SOFTWARE for w in ws) / n
    measurements = len(re.findall(r"\b\d+(?:\.\d+)?\s?(?:mm|cm|m|in|ft|kg|g|lb|oz|°[CF]|volt|amp|psi)\b", text, re.I)) / n
    boilerplate = len(BOILERPLATE_RE.findall(text + "\n" + removed_text)) / n
    repetition = max(0.0, 1.0 - unique * 1.6)
    physical_score = min(1.0, 9 * physical + 6 * procedure + 15 * measurements)
    recoverability = min(1.0, 2.5 * boilerplate + 1.5 * repetition + (0.25 if removed_text else 0))
    quality = max(0.0, min(1.0, math.log10(n + 1) / 3.2 + unique * .35 - repetition - software * 2))
    return {
        "physical_score": round(physical_score, 6), "recoverability_score": round(recoverability, 6),
        "quality_score": round(quality, 6), "physical_keyword_density": physical,
        "procedure_density": procedure, "software_density": software, "measurement_density": measurements,
        "boilerplate_ratio": boilerplate, "repetition_ratio": repetition, "url": url,
    }


def simhash(text: str, bits: int = 64) -> int:
    ws = words(text)
    shingles = [" ".join(ws[i:i + 5]) for i in range(max(1, len(ws) - 4))] or [text]
    vector = [0] * bits
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")
        for i in range(bits):
            vector[i] += 1 if value & (1 << i) else -1
    return sum((1 << i) for i, value in enumerate(vector) if value >= 0)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class SimhashIndex:
    """Candidate index: <=3 changed bits guarantees a match in one of four bands."""
    def __init__(self, max_distance: int = 3, bands: int = 4):
        if max_distance >= bands:
            raise ValueError("bands must exceed max_distance")
        self.max_distance = max_distance
        self.bands = bands
        self.width = 64 // bands
        self.mask = (1 << self.width) - 1
        self.buckets: dict[tuple[int, int], list[int]] = {}

    def is_near(self, value: int) -> bool:
        candidates: set[int] = set()
        for band in range(self.bands):
            candidates.update(self.buckets.get((band, (value >> (band * self.width)) & self.mask), ()))
        return any(hamming(value, other) <= self.max_distance for other in candidates)

    def add(self, value: int) -> None:
        for band in range(self.bands):
            key = (band, (value >> (band * self.width)) & self.mask)
            self.buckets.setdefault(key, []).append(value)


def ngrams(text: str, n: int) -> set[str]:
    ws = words(text)
    return {" ".join(ws[i:i+n]) for i in range(max(0, len(ws)-n+1))}


def piqa_overlap(text: str, signatures: set[str], n: int = 8) -> float:
    grams = ngrams(text, n)
    return len(grams & signatures) / max(1, min(len(grams), 8))


def risk_flags(text: str, source_type: str = "") -> dict[str, bool]:
    flags = {name: bool(pattern.search(text)) for name, pattern in RISK_PATTERNS.items()}
    ws = words(text)
    flags["software_only"] = sum(w in SOFTWARE for w in ws) > max(2, 2 * sum(w in PHYSICAL for w in ws))
    flags["pure_marketing"] = bool(re.search(r"\b(?:buy now|best price|limited offer|shop now)\b", text, re.I)) and not any(w in PHYSICAL for w in ws)
    flags["doubtful_reliability"] = False
    return flags


def parse_bool(value: object) -> bool:
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)): return value != 0
    if isinstance(value, str): return value.strip().lower() in {"true", "1", "yes"}
    return False


def classify(record: dict, cfg: dict) -> str:
    judge = record.get("recall_judge") or record.get("precision_judge") or {}
    flags = judge.get("risk_flags") or risk_flags(record.get("text_clean", ""), judge.get("source_type", ""))
    if any(parse_bool(v) for v in flags.values()): return "bad"
    relevance = int(judge.get("task_relevance", 0)); transferable = int(judge.get("transferable_knowledge", 0))
    useful = record.get("useful_tokens", 0); noise = record.get("semantic_noise_fraction", 1.0)
    if relevance < 2 or transferable < 2 or useful < cfg["min_useful_tokens"]: return "bad"
    if noise <= cfg["gem_max_noise"]: return "gem"
    if cfg["hidden_gem_min_noise"] <= noise <= cfg["hidden_gem_max_noise"]: return "hidden_gem"
    return "uncertain"


def read_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True); count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n"); count += 1
    return count
