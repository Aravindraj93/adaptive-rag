"""Audit frozen selection, query sampling, and evaluation invariants."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from conservative_policy import calibrate


def verify(root, development):
    policy_file = root/'phase11-policy.json'
    sample_file = root/'phase11-sample.json'
    policy = json.loads(policy_file.read_text())
    payload = development.read_bytes()
    assert policy['development_sha256'] == hashlib.sha256(payload).hexdigest()
    reproduced = calibrate(json.loads(payload))
    assert all(policy[k] == v for k,v in reproduced.items())
    assert not set(policy['fit_ids']) & set(policy['guard_ids'])
    sample = json.loads(sample_file.read_text())
    report = json.loads((root/'benchmark-phase11-arguana.json').read_text())
    fingerprint = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    assert report['policy_sha256'] == sample['policy_sha256'] == fingerprint
    assert report['sample_sha256'] == hashlib.sha256(sample_file.read_bytes()).hexdigest()
    assert sample['provenance'] == report['provenance']
    assert len(sample['query_ids']) == len(set(sample['query_ids'])) == 200
    assert report['documents'] == 8674 and report['total_dataset_queries'] == 1406
    for name, summary in report['results'].items():
        rows = summary['per_query']
        assert len(rows) == 400
        for rep in (0,1):
            part = [r for r in rows if r['repetition'] == rep]
            assert len(part) == 200 and {r['query_id'] for r in part} == set(sample['query_ids'])
        for r in rows:
            assert r['query_id'] not in r['ids']
            assert len(r['ids']) <= 10 and len(r['ids']) == len(set(r['ids']))
            assert math.isfinite(r['ms']) and r['ms'] >= 0
            assert 0 <= r['recall'] <= 1 and 0 <= r['ndcg'] <= 1.000000001
            expected = (0 if name == 'bm25' or (name == 'selected' and r['route'] == 'primary') else 1)
            assert r['embedding_calls'] == expected
        for k in ('recall','ndcg','ms','embedding_calls'):
            assert math.isclose(summary[k], statistics.fmean(r[k] for r in rows))
    if policy['selected']['threshold'] is None:
        a = report['results']['fusion']['per_query']
        b = report['results']['selected']['per_query']
        assert [r['ids'] for r in a] == [r['ids'] for r in b]
        assert all(r['route'] == 'fusion' and not r['reused_primary'] for r in b)
    print('Calibration replay, policy/sample hashes, self exclusion, counts and metrics: PASS')


if __name__ == '__main__':
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path('phase11-results'),
           Path('phase10-results/phase10-development.json'))
