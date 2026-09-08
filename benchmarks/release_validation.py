"""Bounded release checks: multi-domain replay, concurrent serving, index lifecycle."""
import argparse
import concurrent.futures
import json
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from adaptive_rag import (BM25Retriever, CachedRetriever, Chunk, Query, SearchResult,
                          SegmentedBM25Index, SegmentedBM25Retriever)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = ['benchmark-phase9-cranfield-heldout.json', 'benchmark-phase9-scifact-transfer.json',
               'phase10-results/benchmark-phase10-nfcorpus.json',
               'phase11-results/benchmark-phase11-arguana.json']
    domains = []
    for source in sources:
        data = json.loads((root/source).read_text())
        rows = {str(r['query_id']): r for r in data['results']['fusion']['per_query']}
        class Replay:
            calls = 0
            def search(self, query, *, top_k=10):
                self.calls += 1
                r = rows[query]
                ids = r.get('ids', r.get('result_ids'))
                return [SearchResult(Chunk(d,d,'recorded ranking'), 1/i, i, 'replay')
                        for i,d in enumerate(ids[:top_k], 1)]
        backend = Replay()
        cached = CachedRetriever(backend, revision=lambda: 1, scope=source, max_entries=1000)
        matches = 0
        for qid in rows:
            expected = rows[qid].get('ids', rows[qid].get('result_ids'))
            for _ in range(3):
                actual = [r.chunk.id for r in cached.search(qid, top_k=10)]
                if actual != expected:
                    raise AssertionError('cached ranking changed')
                matches += 1
        domains.append(dict(source=source, unique_queries=len(rows), identical_rankings=matches,
                            backend_calls=backend.calls, cache=asdict(cached.info())))
    corpus = [Chunk(str(i),str(i), f'category{i%100} item{i} shared generic text') for i in range(10000)]
    backend = BM25Retriever()
    backend.add(corpus)
    cached = CachedRetriever(backend, revision=lambda: 1, scope='synthetic-soak', max_entries=128)
    expected = {i:[r.chunk.id for r in backend.search(f'category{i}', top_k=10)] for i in range(100)}
    started = time.perf_counter()
    def search(i):
        q = i%100
        assert [r.chunk.id for r in cached.search(f'category{q}', top_k=10)] == expected[q]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(search, range(20000)))
    soak = dict(documents=10000, searches=20000, workers=8,
                elapsed_seconds=time.perf_counter()-started, cache=asdict(cached.info()),
                retained_backend_telemetry=len(backend.telemetry.snapshot()))
    with tempfile.TemporaryDirectory() as folder:
        manager = SegmentedBM25Index.create(Path(folder)/'index', [Chunk('base','base','shared text')])
        for i in range(30):
            manager.append([Chunk(f'n{i}',f'n{i}','shared update')])
            if i:
                manager.delete([f'n{i-1}'])
            with SegmentedBM25Retriever(manager.path) as pinned:
                old = {c.id for c in pinned.iter_chunks()}
                manager.compact()
                manager.cleanup(dry_run=False)
                assert {c.id for c in pinned.iter_chunks()} == old
            manager.cleanup(dry_run=False)
            with SegmentedBM25Retriever(manager.path) as reader:
                assert {c.id for c in reader.iter_chunks()} == {'base',f'n{i}'}
    report = dict(replay_note='Previously observed public rankings; correctness replay, not new quality or latency evidence',
                  multi_domain=domains, concurrent_serving=soak,
                  index_lifecycle=dict(cycles=30, reader_preservation=True, final_live_documents=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
