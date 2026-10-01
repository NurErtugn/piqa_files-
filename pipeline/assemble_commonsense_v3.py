"""Assemble validated rewrites with conservative grounded fallbacks."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/capstor/scratch/cscs/nertug/fineweb-gems/production10m')
BASE = ROOT / 'commonsense-v3'
DEST = BASE / 'assembled-predecontam'


def records(path):
    if not path.exists():
        raise RuntimeError(f'Missing file: {path}')
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    if DEST.exists():
        raise RuntimeError(f'{DEST} already exists; inspect rather than overwrite')
    DEST.mkdir()
    seen_all = set()
    counts = Counter()
    for shard in range(79):
        input_path = BASE / 'input' / f'part-{shard}.jsonl'
        validation_path = BASE / 'validation' / f'part-{shard}.jsonl'
        error_path = BASE / 'errors' / f'part-{shard}.jsonl'
        output_path = DEST / f'part-{shard}.jsonl'
        source = {row['id']: row for row in records(input_path)}
        validated = {row['id']: row for row in records(validation_path)}
        errors = {row['id']: row for row in records(error_path)}
        if len(source) != sum(1 for _ in records(input_path)):
            raise RuntimeError(f'Duplicate input IDs in shard {shard}')
        if len(validated) != sum(1 for _ in records(validation_path)):
            raise RuntimeError(f'Duplicate validated IDs in shard {shard}')
        if len(errors) != sum(1 for _ in records(error_path)):
            raise RuntimeError(f'Duplicate error IDs in shard {shard}')
        if set(validated) & set(errors) or set(source) != set(validated) | set(errors):
            raise RuntimeError(f'Input/validation/error mismatch in shard {shard}')
        with output_path.open('x', encoding='utf-8') as output:
            for record_id, original in source.items():
                if record_id in seen_all:
                    raise RuntimeError(f'Duplicate ID across shards: {record_id}')
                seen_all.add(record_id)
                if original['passage_review_status'] != 'passage_review_pass':
                    raise RuntimeError(f'Unapproved passage: {record_id}')
                grounded = original['training_text'].strip()
                if not grounded:
                    raise RuntimeError(f'Empty grounded passage: {record_id}')
                if record_id in validated:
                    row = validated[record_id]
                    if row['grounded_extraction'] != grounded:
                        raise RuntimeError(f'Grounding changed: {record_id}')
                    verdict = row['commonsense_validation']
                    if verdict['decision'] == 'pass':
                        required = (not verdict['unsupported'] and not verdict['misleading']
                                    and not verdict['incomplete']
                                    and verdict['contains_useful_physical_lesson']
                                    and not verdict['uncertain'])
                        if not required:
                            raise RuntimeError(f'Invalid pass: {record_id}')
                        final = row['commonsense_training_candidate'].strip()
                        if not final:
                            raise RuntimeError(f'Empty approved rewrite: {record_id}')
                        selection = 'validated_commonsense_rewrite'
                    elif verdict['decision'] == 'fallback':
                        final = grounded
                        selection = 'grounded_fallback_validation'
                    else:
                        raise RuntimeError(f'Unknown validation decision: {record_id}')
                else:
                    row = original
                    final = grounded
                    selection = 'grounded_fallback_generation_error'
                    row['rewrite_technical_error'] = errors[record_id]
                row['final_training_text'] = final
                row['final_selection'] = selection
                row['final_status'] = 'predecontam_candidate'
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[selection] += 1
                counts[row['candidate_category']] += 1
                counts['total'] += 1
        print(f'Assembled {shard + 1}/79; records {counts["total"]}', flush=True)
    if counts['total'] != 78622 or len(seen_all) != 78622:
        raise RuntimeError(f'Unexpected totals: {dict(counts)}')
    manifest = {'records': counts['total'], 'unique_ids': len(seen_all),
                'counts': dict(counts), 'shards': 79,
                'status': 'predecontam_candidate',
                'note': 'Not final until contamination and duplicate checks complete.'}
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
