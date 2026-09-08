"""Pre-frozen policy transfer to a deterministic ArguAna sample, full corpus."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
import zipfile
from pathlib import Path

from adaptive_rag import (AdaptiveFusionRetriever, FusionPolicy, BM25Retriever,
                          DenseRetriever, Chunk, Query, SearchResult,
                          ReciprocalRankFusionRetriever)
from adaptive_rag.hardware import profile_hardware
from evaluate_routing import Encoder, metrics


class ExcludeSelf:
    """Evaluation-only self exclusion before fusion; refill depth and rerank."""
    def __init__(self, backend):
        self.backend = backend

    def search(self, query, *, top_k=10):
        qid = query.metadata['evaluation_query_id']
        candidates = self.backend.search(query, top_k=top_k+1)
        kept = [r for r in candidates if r.chunk.id != qid][:top_k]
        return [SearchResult(r.chunk, r.score, i, r.source) for i, r in enumerate(kept, 1)]


def main():
    parser = argparse.ArgumentParser()
    for name in ('assets', 'dataset', 'work', 'policy', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    destination = args.output/'benchmark-phase11-arguana.json'
    if destination.exists():
        raise ValueError('Use a new output directory; never overwrite a recorded test')
    policy_bytes = args.policy.read_bytes()
    policy_hash = hashlib.sha256(policy_bytes).hexdigest()
    policy = json.loads(policy_bytes)
    selected = policy['selected']
    provenance = json.loads((args.assets/'provenance.json').read_text())
    for name, record in provenance['files'].items():
        if hashlib.sha256((args.assets/name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('input checksum mismatch: '+name)
    payload = args.dataset.read_bytes()
    if hashlib.md5(payload).hexdigest() != '8ad3e3c2a5867cdced806d6503f29b99':
        raise ValueError('ArguAna differs from the BEIR published archive')
    provenance['arguana'] = dict(sha256=hashlib.sha256(payload).hexdigest(),
        url='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip')
    with zipfile.ZipFile(args.dataset) as archive:
        corpus = [json.loads(line) for line in archive.read('arguana/corpus.jsonl').splitlines()]
        queries = {d['_id']: d['text'] for d in
                   map(json.loads, archive.read('arguana/queries.jsonl').splitlines())}
    assert len(corpus) == 8674 and len(queries) == 1406
    sample = sorted(queries, key=lambda q: hashlib.sha256(('phase11-sample:'+q).encode()).hexdigest())[:200]
    manifest = dict(query_ids=sample, total_queries=len(queries), sample_size=200,
                    rule='first 200 SHA256(phase11-sample:query_id) ordered IDs',
                    policy_sha256=policy_hash, provenance=provenance)
    sample_path = args.output/'phase11-sample.json'
    if sample_path.exists():
        if json.loads(sample_path.read_text()) != manifest:
            raise ValueError('sample or policy changed since evaluation started')
    else:
        with sample_path.open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
    # No relevance labels are accessed before the policy and sample are frozen.
    qrels = {}
    with zipfile.ZipFile(args.dataset) as archive:
        for line in archive.read('arguana/qrels/test.tsv').decode().splitlines()[1:]:
            qid, did, grade = line.split()
            if qid in sample:
                qrels.setdefault(qid, {})[did] = int(grade)
    assert set(qrels) == set(sample)
    encoder = Encoder(args.assets)
    key = hashlib.sha256(json.dumps([provenance, 256, 'title+text/masked-mean/l2']).encode()).hexdigest()
    cache = args.work/(key+'.json')
    chunks = [Chunk(d['_id'], d['_id'], d.get('title', '')+' '+d['text']) for d in corpus]
    if cache.exists():
        vectors = json.loads(cache.read_text())
    else:
        vectors = []
        for start in range(0, len(chunks), 16):
            vectors.extend(encoder([c.text for c in chunks[start:start+16]]))
            if start % 640 == 0:
                print('Corpus embedding', start, '/', len(chunks), flush=True)
        cache.write_text(json.dumps(vectors))
    by_text = {c.text: v for c,v in zip(chunks,vectors)}
    dense = DenseRetriever(lambda texts: [by_text[t] for t in texts])
    dense.add(chunks)
    dense.embedder = encoder
    sparse = BM25Retriever()
    sparse.add(chunks)
    lexical = ExcludeSelf(sparse)
    fusion = ReciprocalRankFusionRetriever([lexical, ExcludeSelf(dense)])
    router = AdaptiveFusionRetriever(fusion, policy=FusionPolicy(selected['threshold'], selected['weights']))
    modes = dict(bm25=lexical, fusion=fusion, selected=router)
    observations = {mode: [] for mode in modes}
    encoder(['Warm up the CPU embedding model'])
    for repetition in range(2):
        for i, qid in enumerate(sample):
            query = Query(queries[qid], {'evaluation_query_id': qid})
            names = list(modes)
            offset = (i+repetition) % len(names)
            for name in names[offset:]+names[:offset]:
                before = encoder.calls
                started = time.perf_counter()
                ranked = modes[name].search(query, top_k=10)
                elapsed = (time.perf_counter()-started)*1000
                ids = [r.chunk.id for r in ranked]
                assert qid not in ids
                row = dict(query_id=qid, repetition=repetition, ids=ids, ms=elapsed,
                           embedding_calls=encoder.calls-before, **metrics(ids, qrels[qid]))
                if name == 'selected':
                    row.update(route=router.last_decision.route,
                               reused_primary=router.last_decision.reused_primary)
                observations[name].append(row)
            if i % 25 == 0:
                print('Test', repetition+1, i, '/ 200', flush=True)
    result = {}
    for name, rows in observations.items():
        times = sorted(r['ms'] for r in rows)
        result[name] = {k: statistics.fmean(r[k] for r in rows)
                        for k in ('recall', 'ndcg', 'ms', 'embedding_calls')}
        result[name].update(median_ms=statistics.median(times),
                            p95_ms=times[math.ceil(.95*len(times))-1], per_query=rows)
    report = dict(dataset='ArguAna fixed 200-query sample', documents=len(chunks),
        queries=200, total_dataset_queries=1406, repetitions=2, top_k=10,
        policy_sha256=policy_hash, sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        provenance=provenance, hardware=profile_hardware().to_dict(),
        packages={p: importlib.metadata.version(p) for p in ('numpy','onnxruntime','tokenizers')},
        timing='Warm search including uncached query embedding, excluding indexing and startup',
        self_exclusion='Each backend requests depth+1, removes identical query/document ID, reranks before RRF',
        results=result)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print({n:{k:v for k,v in r.items() if k != 'per_query'} for n,r in result.items()}, flush=True)


if __name__ == '__main__':
    main()
