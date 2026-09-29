"""Paired before/after retrieval evaluation; no LLM or hosted service required."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

from .benchmark import load_dataset, sample_dataset
from .hardware import profile_hardware
from .models import Query


@dataclass(frozen=True)
class ComparisonCase:
    query: str | Query
    relevant_chunk_ids: frozenset[str] | None = None


def _quality(ids, relevant, top_k):
    if relevant is None or not relevant:
        return None
    matches = [i for i, key in enumerate(ids, start=1) if key in relevant]
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(top_k, len(relevant)) + 1))
    return dict(recall=len(matches)/len(relevant),
                ndcg=sum(1/math.log2(i+1) for i in matches)/ideal)


def compare_retrievers(baseline, candidate, cases, *, top_k=5, repetitions=1):
    """Alternate baseline/candidate order; return timings and ranking regressions.

    Pass independent, equivalently configured backends with identical corpora.
    Results are measured, not assumed equivalent. Repetitions deliberately reuse
    queries; first-pass and subsequent-pass latencies are separate. Index builds,
    RAM, API costs and embedding counts are NOT measured by this generic harness.
    Caller controls warm state; first pass is not necessarily a cold process.
    Reports contain chunk IDs but omit query/document text and metadata.
    """
    if any(type(v) is not int or v < 1 for v in (top_k, repetitions)):
        raise ValueError('positive integer top_k and repetitions required')
    cases = list(cases)
    if not cases:
        raise ValueError('at least one query required')
    records = []
    for repeat in range(repetitions):
        for index, case in enumerate(cases):
            order = [('baseline', baseline), ('candidate', candidate)]
            if (repeat * len(cases) + index) % 2:
                order.reverse()
            record = dict(case=index, repetition=repeat)
            for name, backend in order:
                start = perf_counter()
                rows = list(backend.search(case.query, top_k=top_k))
                duration = (perf_counter() - start) * 1000
                ids = [row.chunk.id for row in rows]
                if len(ids) > top_k or len(ids) != len(set(ids)):
                    raise ValueError('backend returned too many results or duplicate chunk IDs')
                record[name] = dict(ids=ids, latency_ms=duration,
                                    quality=_quality(ids, case.relevant_chunk_ids, top_k))
            record['ranking_changed'] = record['baseline']['ids'] != record['candidate']['ids']
            bq, cq = record['baseline']['quality'], record['candidate']['quality']
            record['quality_regression'] = None if bq is None else (
                cq['recall'] < bq['recall'] or cq['ndcg'] < bq['ndcg'])
            records.append(record)
    def timings(values):
        if not values:
            return None
        return dict(median_ms=statistics.median(values),
                    p95_ms=sorted(values)[math.ceil(.95 * len(values))-1])
    summaries = {}
    for name in ('baseline', 'candidate'):
        qualities = [r[name]['quality'] for r in records if r[name]['quality'] is not None]
        summaries[name] = dict(
            latency=timings([r[name]['latency_ms'] for r in records]),
            first_pass=timings([r[name]['latency_ms'] for r in records if r['repetition'] == 0]),
            repeated_passes=timings([r[name]['latency_ms'] for r in records if r['repetition'] > 0]),
            recall_at_k=statistics.fmean(q['recall'] for q in qualities) if qualities else None,
            ndcg_at_k=statistics.fmean(q['ndcg'] for q in qualities) if qualities else None)
    return dict(schema_version=1, top_k=top_k, unique_cases=len(cases),
                executions_per_backend=len(records), repetitions=repetitions,
                labelled_cases=sum(bool(c.relevant_chunk_ids) for c in cases),
                ranking_changes=sum(r['ranking_changed'] for r in records),
                quality_regressions=sum(r['quality_regression'] is True for r in records),
                quality_evaluated=any(bool(c.relevant_chunk_ids) for c in cases),
                summary=summaries, per_query=records,
                limitations=['Binary relevance judgments; unlabelled/empty judgments excluded.',
                             'Repeated queries are intentional; results are workload-specific.',
                             'No RAM, indexing, dollar-cost or embedding-call measurements.',
                             'Ranking agreement is not proof of relevance or answer correctness.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare BM25 without/with bounded exact caching')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args(argv)
    from .retrievers.bm25 import BM25Retriever
    from .semantic_cache import SemanticCachedRetriever
    if args.dataset:
        name, chunks, cases = load_dataset(args.dataset)
    else:
        name = 'six-query-smoke-not-a-performance-claim'
        chunks, cases = sample_dataset()
    baseline, backend = BM25Retriever(), BM25Retriever()
    baseline.add(chunks)
    backend.add(chunks)
    candidate = SemanticCachedRetriever(backend, revision=lambda: 0, scope='comparison')
    report = compare_retrievers(baseline, candidate, cases, top_k=args.top_k,
                               repetitions=args.repetitions)
    from dataclasses import asdict
    report.update(dataset=name, hardware=profile_hardware().to_dict(),
                  candidate_cache=asdict(candidate.info()))
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
