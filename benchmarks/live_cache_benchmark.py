"""Actual CPU inference timing on an explicitly repeated, previously seen workload."""
import argparse
import hashlib
import json
import statistics
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from adaptive_rag import BM25Retriever, DenseRetriever, Chunk, CachedRetriever, ReciprocalRankFusionRetriever
from evaluate_routing import Encoder


def main():
    parser = argparse.ArgumentParser()
    for name in ('assets','data','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    previous = json.loads((root/'phase10-results/benchmark-phase10-nfcorpus.json').read_text())
    provenance = previous['provenance']
    for name, item in provenance['files'].items():
        assert hashlib.sha256((args.assets/name).read_bytes()).hexdigest() == item['sha256']
    archive_path = args.data/'nfcorpus.zip'
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == provenance['nfcorpus']['sha256']
    key = hashlib.sha256(json.dumps([provenance,256,'masked-mean-l2']).encode()).hexdigest()
    vectors = json.loads((args.data/(key+'.json')).read_text())
    with zipfile.ZipFile(archive_path) as archive:
        corpus = list(map(json.loads, archive.read('nfcorpus/corpus.jsonl').splitlines()))
        queries = {d['_id']:d['text'] for d in map(json.loads, archive.read('nfcorpus/queries.jsonl').splitlines())}
    chunks = [Chunk(d['_id'], d['_id'], d.get('title','')+' '+d['text']) for d in corpus]
    by_text = {c.text:v for c,v in zip(chunks,vectors)}
    encoder = Encoder(args.assets)
    dense = DenseRetriever(lambda texts: [by_text[t] for t in texts])
    dense.add(chunks)
    dense.embedder = encoder
    sparse = BM25Retriever()
    sparse.add(chunks)
    fusion = ReciprocalRankFusionRetriever([sparse,dense])
    cached = CachedRetriever(fusion, revision=lambda: 'nfcorpus/model-pinned/config-v1', scope='benchmark')
    ids = sorted({r['query_id'] for r in previous['results']['fusion']['per_query']})[:20]
    encoder(['Warm up CPU inference'])
    results = dict(fusion=[], cached=[])
    exact = 0
    for repetition in range(5):
        for i,qid in enumerate(ids):
            outputs = {}
            for mode in (['fusion','cached'] if (i+repetition)%2 else ['cached','fusion']):
                backend = fusion if mode == 'fusion' else cached
                before = encoder.calls
                started = time.perf_counter()
                outputs[mode] = backend.search(queries[qid], top_k=10)
                elapsed = (time.perf_counter()-started)*1000
                results[mode].append(dict(query_id=qid, repetition=repetition, ms=elapsed,
                                          embedding_calls=encoder.calls-before))
            assert outputs['fusion'] == outputs['cached']
            exact += 1
        print('Completed repetition', repetition+1, flush=True)
    summary = {mode:dict(mean_ms=statistics.fmean(r['ms'] for r in rows),
                         embedding_calls=sum(r['embedding_calls'] for r in rows))
               for mode,rows in results.items()}
    report = dict(protocol='20 previously observed NFCorpus queries, five repetitions, alternating mode order; full 3633-document corpus; warm model, live uncached query embeddings on misses',
                  exact_result_matches=exact, queries_per_mode=100, unique_queries=20,
                  summary=summary, cache=asdict(cached.info()), observations=results,
                  provenance=provenance,
                  limitation='Designed 80% repeat rate; no inference saving on novel queries; revision contract required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(summary, 'exact matches', exact, flush=True)


if __name__ == '__main__':
    main()
