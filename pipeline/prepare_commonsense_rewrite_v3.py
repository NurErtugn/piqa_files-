"""Build immutable rewrite input from passage-review passes only."""
import json
from pathlib import Path

ROOT = Path('/capstor/scratch/cscs/nertug/fineweb-gems/production10m')
SOURCE = ROOT / 'passage-risk-v3/output'
DEST = ROOT / 'commonsense-v3/input'


def main():
    if DEST.exists():
        raise RuntimeError(f'{DEST} already exists; inspect before rerunning')
    DEST.mkdir(parents=True)
    seen = set()
    total = 0
    output = None
    try:
        for shard in range(97):
            path = SOURCE / f'part-{shard}.jsonl'
            with path.open(encoding='utf-8') as source:
                for line in source:
                    row = json.loads(line)
                    if row['passage_review_status'] != 'passage_review_pass':
                        continue
                    if row['id'] in seen:
                        raise RuntimeError(f'Duplicate ID: {row["id"]}')
                    seen.add(row['id'])
                    if total % 1000 == 0:
                        if output is not None:
                            output.close()
                        output = (DEST / f'part-{total // 1000}.jsonl').open('x', encoding='utf-8')
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    total += 1
            print(f'Checked {shard + 1}/97; rewrite inputs {total}', flush=True)
    finally:
        if output is not None:
            output.close()
    if total != 78622:
        raise RuntimeError(f'Expected 78622 passes; got {total}. Do not submit rewrite.')
    manifest = {'records': total, 'unique_ids': len(seen),
                'shard_size': 1000, 'output_shards': (total + 999) // 1000,
                'source': str(SOURCE), 'status': 'rewrite_input_ready',
                'note': 'Passing verbatim passages only; quarantine remains untouched.'}
    (ROOT / 'commonsense-v3/input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
