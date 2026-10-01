"""Extract review candidates; never label these passages training-ready."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/capstor/scratch/cscs/nertug/fineweb-gems/production10m')
DEST = ROOT / 'selected-passages-v3'


def main():
    if DEST.exists():
        raise RuntimeError(f'{DEST} already exists; inspect instead of overwriting')
    DEST.mkdir()
    (DEST / 'input').mkdir()
    counts = Counter()
    risks = Counter()
    ids = set()
    output = None
    try:
        for shard in range(169):
            source = ROOT / 'independent-v3' / f'part-{shard}.jsonl'
            with source.open(encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    judge = row['independent_judge']
                    counts['audited'] += 1
                    if judge['decision'] != 'useful':
                        continue
                    indices = judge['useful_segments']
                    segments = row['segments']
                    if (judge.get('validation_flags') or not indices
                            or min(judge[k] for k in ('physical_relevance', 'transferability', 'reliability')) < 2
                            or any(type(i) is not int or not 0 <= i < len(segments) for i in indices)):
                        raise RuntimeError(f'Invalid useful candidate: {row["id"]}')
                    if row['id'] in ids:
                        raise RuntimeError(f'Duplicate ID: {row["id"]}')
                    ids.add(row['id'])
                    selected = [{'index': i, 'text': segments[i]} for i in sorted(set(indices))]
                    text = '\n\n'.join(x['text'] for x in selected).strip()
                    if not text:
                        raise RuntimeError(f'Empty passage: {row["id"]}')
                    combined = {}
                    for annotation in (row.get('precision_judge', {}), judge):
                        for key, value in annotation.get('risk_flags', {}).items():
                            if type(value) is not bool:
                                raise RuntimeError(f'Nonboolean risk flag: {row["id"]}')
                            combined[key] = combined.get(key, False) or value
                    risks.update(k for k, v in combined.items() if v)
                    n = counts['selected']
                    if n % 1000 == 0:
                        if output is not None:
                            output.close()
                        output = (DEST / 'input' / f'part-{n // 1000}.jsonl').open('x', encoding='utf-8')
                    candidate = {
                        'id': row['id'], 'url': row.get('url'),
                        'candidate_category': judge['category'],
                        'training_text': text, 'selected_segments': selected,
                        'source_reference': {'path': str(source), 'line': line_number},
                        'precision_judge': row.get('precision_judge', {}),
                        'independent_judge': judge,
                        'independent_judge_sampling': row.get('independent_judge_sampling', {}),
                        'combined_document_risk_flags': combined,
                        'status': 'selected_passage_pending_risk_review',
                    }
                    output.write(json.dumps(candidate, ensure_ascii=False) + '\n')
                    counts['selected'] += 1
                    counts[judge['category']] += 1
            print(f'Checked {shard + 1}/169 shards; selected {counts["selected"]}', flush=True)
    finally:
        if output is not None:
            output.close()
    if counts['audited'] != 420842 or counts['selected'] != 96977:
        raise RuntimeError(f'Unexpected totals: {dict(counts)}; outputs are not ready')
    summary = {'counts': dict(counts), 'unique_ids': len(ids),
               'document_risk_flags': dict(risks), 'shard_size': 1000,
               'output_shards': (counts['selected'] + 999) // 1000,
               'status': 'pending_passage_level_risk_review',
               'note': 'Verbatim selected passages; document risk flags are preserved, not automatically removed.'}
    (DEST / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
