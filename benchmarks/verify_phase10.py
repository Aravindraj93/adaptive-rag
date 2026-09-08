"""Independent, dependency-free audit of the saved Phase 10 experiment."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def verify(root):
    policy_path = root/'phase10-policy.json'
    policy = json.loads(policy_path.read_text())
    dev = json.loads((root/'phase10-development.json').read_text())
    report = json.loads((root/'benchmark-phase10-nfcorpus.json').read_text())
    assert report['policy_sha256'] == hashlib.sha256(policy_path.read_bytes()).hexdigest()
    assert report['provenance'] == policy['provenance']
    assert len(dev) == len(policy['development_ids']) == report['development_queries']
    assert {r['query_id'] for r in dev} == set(policy['development_ids'])
    assert policy['selected'] == min((c for c in policy['candidates'] if c['feasible']),
                                      key=lambda c: c['ms'])
    for name, summary in report['results'].items():
        rows = summary['per_query']
        assert len(rows) == 646
        for repetition in (0, 1):
            subset = [r for r in rows if r['repetition'] == repetition]
            assert len(subset) == 323 and len({r['query_id'] for r in subset}) == 323
            assert not {r['query_id'] for r in subset} & set(policy['development_ids'])
        for row in rows:
            assert 0 <= row['recall'] <= 1 and 0 <= row['ndcg'] <= 1.000000001
            assert len(row['ids']) <= 10 and len(set(row['ids'])) == len(row['ids'])
            assert math.isfinite(row['ms']) and row['ms'] >= 0
            if name == 'selected':
                assert row['embedding_calls'] == (0 if row['route'] == 'primary' else 1)
                expected_reuse = row['route'] == 'fusion' and policy['selected']['threshold'] is not None
                assert row['reused_primary'] == expected_reuse
            else:
                assert row['embedding_calls'] == (0 if name == 'bm25' else 1)
        for metric in ('recall', 'ndcg', 'ms', 'embedding_calls'):
            assert math.isclose(summary[metric], statistics.fmean(r[metric] for r in rows))
    baseline = report['results']['fusion']
    selected = report['results']['selected']
    print('Report integrity, frozen policy, dev/test separation and live embedding counts: PASS')
    print('Selected-policy recall delta:', selected['recall']-baseline['recall'])
    print('Selected-policy nDCG delta:', selected['ndcg']-baseline['ndcg'])
    print('Selected-policy mean latency reduction:', 1-selected['ms']/baseline['ms'])
    print('Selected-policy embedding-call reduction:', 1-selected['embedding_calls'])


if __name__ == '__main__':
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('phase10-results'))
