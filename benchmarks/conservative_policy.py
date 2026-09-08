"""Offline, dependency-free two-stage selection. Not a statistical guarantee."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def accepts(row, policy):
    return (policy['threshold'] is not None and bool(row['primary']['ids'])
            and sum(w*f for w, f in zip(policy['weights'], row['features'])) >= policy['threshold'])


def summarize(rows, policy):
    choices = [accepts(r, policy) for r in rows]
    result = {k: statistics.fmean(r['primary' if yes else 'fusion'][k]
              for r, yes in zip(rows, choices)) for k in ('recall', 'ndcg')}
    result['ms'] = statistics.fmean(r['fusion']['ms'] if policy['threshold'] is None else
                      r['primary']['ms']+(0 if yes else r['reuse_extra_ms'])
                      for r, yes in zip(rows, choices))
    result['acceptance'] = statistics.fmean(choices)
    return result


def bootstrap_lower(differences, *, seed=1101, repetitions=2000):
    """2.5th percentile of paired bootstrap means (one-sided 97.5% estimate)."""
    if not differences or repetitions < 100:
        raise ValueError('nonempty observations and at least 100 repetitions required')
    if any(not math.isfinite(d) or not -1 <= d <= 1 for d in differences):
        raise ValueError('quality differences must be finite and in [-1, 1]')
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(differences, k=len(differences)))
                   for _ in range(repetitions))
    return means[max(0, math.ceil(.025*repetitions)-1)]


def calibrate(rows):
    for row in rows:
        features = row['features']
        values = [row[mode][k] for mode in ('primary', 'fusion') for k in ('recall', 'ndcg')]
        times = [row['primary']['ms'], row['fusion']['ms'], row['reuse_extra_ms']]
        if (len(features) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1
                                     for x in list(features)+values)
                or any(not math.isfinite(t) or t < 0 for t in times)):
            raise ValueError('development observations contain invalid features, quality or timing')
    if len(rows) < 30 or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('at least 30 uniquely identified development rows required')
    ordered = sorted(rows, key=lambda r: hashlib.sha256(('phase11:'+r['query_id']).encode()).hexdigest())
    split = len(rows)*2//3
    fit, guard = ordered[:split], ordered[split:]
    baseline = dict(threshold=None, weights=[.75, .25, 0])
    floor = summarize(fit, baseline)
    candidates = [dict(**baseline, **floor, feasible=True)]
    for a in range(5):
        for b in range(5-a):
            for threshold in [i/20 for i in range(21)]:
                policy = dict(threshold=threshold, weights=[a/4, b/4, (4-a-b)/4])
                result = summarize(fit, policy)
                candidates.append(dict(**policy, **result,
                    feasible=all(result[k] >= floor[k]-.005 for k in ('recall', 'ndcg'))))
    proposal = min((c for c in candidates if c['feasible']), key=lambda c: c['ms'])
    guard_stats = summarize(guard, proposal)
    guard_baseline = summarize(guard, baseline)
    bounds = {}
    for k in ('recall', 'ndcg'):
        differences = [r['primary' if accepts(r, proposal) else 'fusion'][k]-r['fusion'][k]
                       for r in guard]
        bounds[k] = dict(mean=statistics.fmean(differences), lower=bootstrap_lower(differences))
    approved = (proposal['threshold'] is not None
                and all(b['lower'] >= -.01 for b in bounds.values())
                and guard_stats['ms'] <= .95*guard_baseline['ms'])
    selected = {k: proposal[k] for k in ('threshold', 'weights')} if approved else baseline
    return dict(protocol='Retrospective NFCorpus development split; fit 2/3, guard 1/3; one proposal only',
        fit_ids=[r['query_id'] for r in fit], guard_ids=[r['query_id'] for r in guard],
        fit_margin=.005, guard_tolerance=.01, guard_min_latency_reduction=.05,
        bootstrap_repetitions=2000, bootstrap_seed=1101, candidates=candidates,
        proposal=proposal, guard_bounds=bounds, guard_summary=guard_stats,
        guard_baseline=guard_baseline, approved=approved, selected=selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--development', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    payload = args.development.read_bytes()
    policy = calibrate(json.loads(payload))
    policy['development_sha256'] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(policy, stream, indent=2)
    print('PROPOSAL', policy['proposal'])
    print('GUARD', policy['guard_bounds'])
    print('APPROVED', policy['approved'], 'FROZEN', policy['selected'])


if __name__ == '__main__':
    main()
