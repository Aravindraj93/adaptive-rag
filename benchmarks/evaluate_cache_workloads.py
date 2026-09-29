"""Deterministic synthetic cache workload checks, not a semantic quality benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from adaptive_rag import (BM25Retriever, Chunk, ComparisonCase,
                          SemanticCachedRetriever, compare_retrievers, profile_hardware)


class CountingBackend:
    def __init__(self, backend):
        self.backend = backend
        self.calls = 0

    def search(self, query, *, top_k=5):
        self.calls += 1
        return self.backend.search(query, top_k=top_k)


def evaluate(chunks=10000, queries=100):
    if chunks < queries or queries < 1:
        raise ValueError('require chunks >= queries >= 1')
    corpus = [Chunk(f'item-{i}', f'doc-{i}',
                    f'catalog policy shared guidance topic{i % 100} item{i}')
              for i in range(chunks)]
    reports = {}
    for name, broad, passes in [('unique_selective', False, 1),
                                ('unique_broad', True, 1),
                                ('repeated_selective', False, 5),
                                ('repeated_broad', True, 5)]:
        baseline, optimized = BM25Retriever(), BM25Retriever()
        baseline.add(corpus)
        optimized.add(corpus)
        counted = CountingBackend(optimized)
        cache = SemanticCachedRetriever(counted, revision=lambda: 1, scope=name)
        cases = [ComparisonCase(('catalog policy shared guidance ' if broad else '')
                                + f'item{i}', frozenset({f'item-{i}'}))
                 for i in range(queries)]
        report = compare_retrievers(baseline, cache, cases, top_k=5, repetitions=passes)
        report.update(cache=asdict(cache.info()), backend_calls=counted.calls,
                      designed_repeat_fraction=(passes-1)/passes)
        if report['ranking_changes'] or report['quality_regressions']:
            raise AssertionError(f'exact-cache regression in {name}')
        if counted.calls != queries:
            raise AssertionError(f'unexpected backend call count in {name}')
        reports[name] = report
    return dict(corpus_chunks=chunks, query_cases=queries, hardware=profile_hardware().to_dict(),
                workload='Synthetic fixed-token corpus, no real embeddings or LLM calls.',
                caveat='Not representative of production; broad queries intentionally match every chunk.',
                reports=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks', type=int, default=10000)
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.chunks, args.queries)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, report in result['reports'].items():
        print(json.dumps(dict(workload=name, backend_calls=report['backend_calls'],
                              ranking_changes=report['ranking_changes'],
                              summary=report['summary'])))


if __name__ == '__main__':
    main()
