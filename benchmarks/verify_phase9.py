"""Check saved Phase 9 reports without requiring model dependencies."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def verify(root):
    policy_path = root/'phase9-policy.json'
    policy = json.loads(policy_path.read_text())
    fingerprint = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    development = set(policy['development_ids'])
    heldout = set(policy['heldout_ids'])
    assert len(development) == 75 and len(heldout) == 150
    assert not development & heldout
    assert development | heldout == {str(i) for i in range(1, 226)}
    feasible = [c for c in policy['candidates'] if c['feasible']]
    assert policy['selected'] == min(feasible, key=lambda c: c['estimated_ms'])
    for name, count in [('cranfield-heldout', 150), ('scifact-transfer', 300)]:
        report = json.loads((root/('benchmark-phase9-'+name+'.json')).read_text())
        assert report['policy_sha256'] == fingerprint
        assert report['queries'] == count and report['repetitions'] == 2
        for mode, result in report['results'].items():
            rows = result['per_query']
            assert len(rows) == count*2
            for repeat in (0, 1):
                subset = [r for r in rows if r['repetition'] == repeat]
                assert len(subset) == count
                assert len({r['query_id'] for r in subset}) == count
                if name == 'cranfield-heldout':
                    assert {r['query_id'] for r in subset} == heldout
            for row in rows:
                assert 0 <= row['recall'] <= 1 and 0 <= row['ndcg'] <= 1.000000001
                assert len(row['ids']) <= 10 and len(set(row['ids'])) == len(row['ids'])
                assert math.isfinite(row['ms']) and row['ms'] >= 0
                if mode == 'bm25':
                    assert row['embedding_calls'] == 0
                elif mode == 'fusion' or policy['selected']['threshold'] is None:
                    assert row['embedding_calls'] == 1
                else:
                    assert row['embedding_calls'] in (0, 1)
            for key in ('ms', 'recall', 'ndcg', 'embedding_calls'):
                assert math.isclose(result[key], statistics.fmean(r[key] for r in rows))
        if policy['selected']['threshold'] is None:
            a = report['results']['fusion']['per_query']
            b = report['results']['calibrated']['per_query']
            assert [r['ids'] for r in a] == [r['ids'] for r in b]
        print(name, 'report integrity passed')
    print('Frozen policy and development/held-out separation passed')


if __name__ == '__main__':
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.'))
