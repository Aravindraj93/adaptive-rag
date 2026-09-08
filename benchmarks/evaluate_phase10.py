"""Fresh NFCorpus dev/test experiment with uncached ONNX query embeddings."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
import zipfile
from pathlib import Path

from adaptive_rag import (AdaptiveFusionRetriever, FusionPolicy, BM25Retriever, DenseRetriever,
                          Chunk, ReciprocalRankFusionRetriever)
from adaptive_rag.adaptive_fusion import confidence_features
from adaptive_rag.hardware import profile_hardware
from adaptive_rag.tokenization import NormalizedTokenizer
from evaluate_routing import Encoder, metrics


def choose(rows):
    if not rows:
        raise ValueError('development observations required')
    baseline = {k: statistics.fmean(r['fusion'][k] for r in rows) for k in ('recall', 'ndcg')}
    candidates = [dict(threshold=None, weights=[.75, .25, 0],
                       ms=statistics.fmean(r['fusion']['ms'] for r in rows),
                       acceptance=0, feasible=True, **baseline)]
    # Fixed simplex grid: 15 weight combinations, 21 thresholds, no test tuning.
    for a in range(5):
        for b in range(5-a):
            weights = (a/4, b/4, (4-a-b)/4)
            for threshold in [i/20 for i in range(21)]:
                p = FusionPolicy(threshold, weights)
                accepted = [bool(r['primary']['ids']) and p.confidence(r['features']) >= threshold
                            for r in rows]
                quality = {k: statistics.fmean(r['primary' if yes else 'fusion'][k]
                              for r, yes in zip(rows, accepted)) for k in baseline}
                cost = statistics.fmean(r['primary']['ms']+(0 if yes else r['reuse_extra_ms'])
                                        for r, yes in zip(rows, accepted))
                candidates.append(dict(threshold=threshold, weights=list(weights), ms=cost,
                    acceptance=statistics.fmean(accepted),
                    feasible=all(quality[k] >= baseline[k]-.01 for k in baseline), **quality))
    return dict(selected=min((c for c in candidates if c['feasible']), key=lambda c: c['ms']),
                tolerance=.01, candidates=candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    policy_path = args.output/'phase10-policy.json'
    if policy_path.exists():
        raise ValueError('Use a new output directory; never overwrite a frozen policy')
    provenance = json.loads((args.assets/'provenance.json').read_text())
    for name, details in provenance['files'].items():
        if hashlib.sha256((args.assets/name).read_bytes()).hexdigest() != details['sha256']:
            raise ValueError('model/input checksum mismatch: '+name)
    payload = args.dataset.read_bytes()
    if hashlib.md5(payload).hexdigest() != 'a89dba18a62ef92f7d323ec890a0d38d':
        raise ValueError('NFCorpus differs from the BEIR published archive')
    provenance['nfcorpus'] = dict(sha256=hashlib.sha256(payload).hexdigest(),
        url='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip')
    def qrels(split):
        with zipfile.ZipFile(args.dataset) as archive:
            lines = archive.read('nfcorpus/qrels/'+split+'.tsv').decode().splitlines()[1:]
        result = {}
        for line in lines:
            qid, did, grade = line.split()
            result.setdefault(qid, {})[did] = max(0, int(grade))
        return result
    with zipfile.ZipFile(args.dataset) as archive:
        docs = [json.loads(line) for line in archive.read('nfcorpus/corpus.jsonl').splitlines()]
        queries = {d['_id']: d['text'] for d in
                   map(json.loads, archive.read('nfcorpus/queries.jsonl').splitlines())}
    dev = qrels('dev')
    chunks = [Chunk(d['_id'], d['_id'], d.get('title', '')+' '+d['text']) for d in docs]
    encoder = Encoder(args.assets)
    key = hashlib.sha256(json.dumps([provenance, 256, 'masked-mean-l2']).encode()).hexdigest()
    cache = args.work/(key+'.json')
    if cache.exists():
        vectors = json.loads(cache.read_text())
    else:
        vectors = []
        for start in range(0, len(chunks), 16):
            vectors.extend(encoder([c.text for c in chunks[start:start+16]]))
            if start % 320 == 0:
                print('Corpus embedding', start, '/', len(chunks), flush=True)
        cache.write_text(json.dumps(vectors))
    by_text = {c.text: v for c, v in zip(chunks, vectors)}
    dense = DenseRetriever(lambda texts: [by_text[t] for t in texts])
    dense.add(chunks)
    dense.embedder = encoder
    lexical = BM25Retriever()
    lexical.add(chunks)
    fusion = ReciprocalRankFusionRetriever([lexical, dense])
    tokenizer = NormalizedTokenizer()
    encoder(['Warm up CPU inference'])

    def timed(call, grades):
        before = encoder.calls
        start = time.perf_counter()
        results = call()
        elapsed = (time.perf_counter()-start)*1000
        ids = [r.chunk.id for r in results]
        return dict(ids=ids, ms=elapsed, embedding_calls=encoder.calls-before,
                    **metrics(ids, grades))

    observations = []
    for i, (qid, grades) in enumerate(sorted(dev.items())):
        text = queries[qid]
        row = dict(query_id=qid)
        # Alternate order of full-fusion timing and primary/reuse timing.
        for mode in (['fusion', 'reuse'] if i % 2 else ['reuse', 'fusion']):
            if mode == 'fusion':
                row['fusion'] = timed(lambda: fusion.search(text, top_k=10), grades)
            else:
                start = time.perf_counter()
                candidates = lexical.search(text, top_k=30)
                features = confidence_features(text, candidates, tokenizer)
                primary_ms = (time.perf_counter()-start)*1000
                ids = [r.chunk.id for r in candidates[:10]]
                row['primary'] = dict(ids=ids, ms=primary_ms, **metrics(ids, grades))
                row['features'] = features
                reused = timed(lambda: fusion._search_with_prefetched(
                    text, top_k=10, primary_results=candidates), grades)
                row['reuse_extra_ms'] = reused['ms']
        assert row['fusion']['ids'] == reused['ids']
        observations.append(row)
        if i % 50 == 0:
            print('Development', i, '/', len(dev), flush=True)
    policy = dict(protocol='NFCorpus official dev only; fixed 315-policy grid plus always-fusion',
                  development_ids=sorted(dev), provenance=provenance, **choose(observations))
    (args.output/'phase10-development.json').write_text(json.dumps(observations, indent=2))
    policy_path.write_text(json.dumps(policy, indent=2))
    fingerprint = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    print('FROZEN', policy['selected'], fingerprint, flush=True)

    # First access to test judgments happens only after the policy is frozen.
    test = qrels('test')
    assert not set(test) & set(dev)
    assert len(test) == 323 and len(chunks) == 3633
    selected = policy['selected']
    router = AdaptiveFusionRetriever(fusion, policy=FusionPolicy(selected['threshold'], selected['weights']))
    modes = dict(bm25=lexical, fusion=fusion, selected=router)
    results = {name: [] for name in modes}
    for repetition in range(2):
        for i, (qid, grades) in enumerate(sorted(test.items())):
            names = list(modes)
            offset = (i+repetition) % len(names)
            for name in names[offset:]+names[:offset]:
                row = timed(lambda: modes[name].search(queries[qid], top_k=10), grades)
                row.update(query_id=qid, repetition=repetition)
                if name == 'selected':
                    row.update(route=router.last_decision.route,
                               reused_primary=router.last_decision.reused_primary)
                results[name].append(row)
            if i % 50 == 0:
                print('Test', repetition+1, i, '/', len(test), flush=True)
    summaries = {}
    for name, rows in results.items():
        times = sorted(r['ms'] for r in rows)
        summaries[name] = {k: statistics.fmean(r[k] for r in rows)
                           for k in ('ms', 'recall', 'ndcg', 'embedding_calls')}
        summaries[name].update(median_ms=statistics.median(times),
                              p95_ms=times[math.ceil(.95*len(times))-1], per_query=rows)
    report = dict(dataset='NFCorpus', documents=len(chunks), development_queries=len(dev),
                  test_queries=len(test), repetitions=2, policy_sha256=fingerprint,
                  provenance=provenance, hardware=profile_hardware().to_dict(),
                  packages={p: importlib.metadata.version(p) for p in ('numpy','onnxruntime','tokenizers')},
                  timing='Warm in-process search including uncached query embedding, excludes startup/indexing',
                  results=summaries)
    (args.output/'benchmark-phase10-nfcorpus.json').write_text(json.dumps(report, indent=2))
    print({name: {k:v for k,v in s.items() if k != 'per_query'} for name,s in summaries.items()}, flush=True)


if __name__ == '__main__':
    main()
