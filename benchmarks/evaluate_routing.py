"""Development-only threshold selection, frozen evaluation, live CPU embeddings.

Evaluation dependencies: numpy, onnxruntime, tokenizers. See PHASE9.md.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from adaptive_rag import BM25Retriever, DenseRetriever, Chunk
from adaptive_rag import ReciprocalRankFusionRetriever, EscalatingRetriever
from adaptive_rag.planner import AdaptiveRetriever
from adaptive_rag.hardware import profile_hardware
from evaluate_cranfield import records


def metrics(ids, grades):
    relevant = {key for key, grade in grades.items() if grade > 0}
    ideal = sum((2**g-1)/math.log2(i+2)
                for i, g in enumerate(sorted(grades.values(), reverse=True)[:10]))
    return dict(recall=len(set(ids) & relevant)/len(relevant) if relevant else 0,
                ndcg=sum((2**grades.get(key, 0)-1)/math.log2(i+2)
                         for i, key in enumerate(ids))/ideal if ideal else 0)


def select_policy(rows, tolerance=0.01):
    """Minimize estimated mean cost subject to two development quality floors.

    A None threshold means always-fusion. Costs include the repeated primary
    search on an escalation, matching EscalatingRetriever's implementation.
    """
    if not rows:
        raise ValueError('development rows must not be empty')
    if not 0 <= tolerance <= 1:
        raise ValueError('tolerance must be between zero and one')
    baseline = {k: statistics.fmean(r['fusion'][k] for r in rows)
                for k in ('recall', 'ndcg')}
    candidates = []
    for threshold in [i/20 for i in range(21)] + [None]:
        chosen = [('fusion' if threshold is None or not r['primary']['ids']
                   or r['confidence'] < threshold else 'primary') for r in rows]
        quality = {k: statistics.fmean(r[c][k] for r, c in zip(rows, chosen))
                   for k in baseline}
        cost = statistics.fmean(r[c]['ms'] + (r['primary']['ms']
                    if c == 'fusion' and threshold is not None else 0)
                    for r, c in zip(rows, chosen))
        candidates.append(dict(threshold=threshold, estimated_ms=cost, **quality,
                               feasible=all(quality[k] >= baseline[k]-tolerance
                                            for k in baseline)))
    best = min((c for c in candidates if c['feasible']), key=lambda c: c['estimated_ms'])
    return dict(selected=best, tolerance=tolerance, candidates=candidates)


class Encoder:
    def __init__(self, assets):
        self.tokenizer = Tokenizer.from_file(str(assets/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(assets/'model.onnx'),
                            sess_options=options, providers=['CPUExecutionProvider'])
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        encoded = self.tokenizer.encode_batch(texts)
        arrays = {'input_ids': np.array([e.ids for e in encoded], dtype=np.int64),
                  'attention_mask': np.array([e.attention_mask for e in encoded], dtype=np.int64),
                  'token_type_ids': np.array([e.type_ids for e in encoded], dtype=np.int64)}
        output = self.session.run(None, {i.name: arrays[i.name]
                                 for i in self.session.get_inputs()})[0]
        mask = arrays['attention_mask'][..., None]
        pooled = (output*mask).sum(axis=1)/np.maximum(mask.sum(axis=1), 1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-frozen', action='store_true',
                        help='Reuse the saved policy and Cranfield result; run only transfer')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.assets/'provenance.json').read_text())
    for name, details in provenance['files'].items():
        if hashlib.sha256((args.assets/name).read_bytes()).hexdigest() != details['sha256']:
            raise ValueError('asset checksum mismatch: '+name)
    with tarfile.open(args.assets/'cran.tar.gz') as archive:
        def read(name):
            return archive.extractfile(name).read().decode().replace('\r', '')
        corpus = list(records(read('cran.all.1400')))
        queries = {str(i): q['W'] for i, q in enumerate(records(read('cran.qry')), 1)}
        qrels = {}
        for line in read('cranqrel').splitlines():
            qid, did, grade = line.split()
            qrels.setdefault(qid, {})[did] = max(0, int(grade))
    cran = ([Chunk(d['id'], d['id'], d.get('T', '')+' '+d.get('W', ''))
             for d in corpus], queries, qrels)
    encoder = Encoder(args.assets)

    def setup(name, chunks):
        # Only corpus vectors are cached. Queries always invoke ONNX at search time.
        key = hashlib.sha256(json.dumps([provenance, [(c.id, c.text) for c in chunks],
                                       256, 'masked-mean-l2']).encode()).hexdigest()
        cache = args.work/(name+'-'+key+'.json')
        started = time.perf_counter()
        if cache.exists():
            vectors = json.loads(cache.read_text())
        else:
            vectors = []
            for start in range(0, len(chunks), 16):
                vectors.extend(encoder([c.text for c in chunks[start:start+16]]))
                if start % 320 == 0:
                    print(name, 'corpus embedded', min(start+16, len(chunks)), flush=True)
            cache.write_text(json.dumps(vectors))
        by_text = {c.text: v for c, v in zip(chunks, vectors)}
        building = True
        def embed(texts):
            return [by_text[t] for t in texts] if building else encoder(texts)
        dense = DenseRetriever(embed)
        dense.add(chunks)
        building = False
        lexical = BM25Retriever()
        lexical.add(chunks)
        fusion = ReciprocalRankFusionRetriever([lexical, dense])
        encoder(['Warm up the retrieval model.'])
        print(name, 'index ready', round(time.perf_counter()-started, 2), flush=True)
        return lexical, fusion

    lexical, fusion = setup('cranfield', cran[0])
    ids = sorted(queries, key=int)
    random.Random(42).shuffle(ids)
    dev_ids, test_ids = ids[:75], ids[75:]
    gate = AdaptiveRetriever(lexical, confidence_threshold=0)

    def measure(retriever, text, grades):
        before = encoder.calls
        started = time.perf_counter()
        ranked = retriever.search(text, top_k=10)
        elapsed = (time.perf_counter()-started)*1000
        result_ids = [r.chunk.id for r in ranked]
        return dict(ids=result_ids, ms=elapsed, embedding_calls=encoder.calls-before,
                    **metrics(result_ids, grades))

    policy_path = args.output/'phase9-policy.json'
    if args.resume_frozen:
        policy = json.loads(policy_path.read_text())
        previous = json.loads((args.output/'benchmark-phase9-cranfield-heldout.json').read_text())
        if hashlib.sha256(policy_path.read_bytes()).hexdigest() != previous['policy_sha256']:
            raise ValueError('frozen policy changed since held-out evaluation')
        if previous['provenance'] != provenance:
            raise ValueError('assets changed since held-out evaluation')
        if policy['development_ids'] != dev_ids or policy['heldout_ids'] != test_ids:
            raise ValueError('query split changed since policy calibration')
    else:
        rows = []
        for i, qid in enumerate(dev_ids):
            row = dict(query_id=qid)
            for name in (['primary', 'fusion'] if i % 2 else ['fusion', 'primary']):
                row[name] = measure(gate if name == 'primary' else fusion, queries[qid], qrels[qid])
            row['confidence'] = gate.last_decision.confidence
            rows.append(row)
        policy = dict(protocol='75 Cranfield development queries, seed 42; no test selection',
                      development_ids=dev_ids, heldout_ids=test_ids, **select_policy(rows))
        (args.output/'phase9-development.json').write_text(json.dumps(rows, indent=2))
        policy_path.write_text(json.dumps(policy, indent=2))
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    threshold = policy['selected']['threshold']
    print('FROZEN POLICY', policy['selected'], policy_hash, flush=True)

    def evaluate(name, data, query_ids, lexical, fusion):
        modes = {'bm25': lexical, 'fusion': fusion,
                 'calibrated': fusion if threshold is None else EscalatingRetriever(
                     lexical, fusion, confidence_threshold=threshold)}
        observations = {mode: [] for mode in modes}
        for repetition in range(2):
            for i, qid in enumerate(query_ids):
                order = list(modes)
                offset = (i+repetition) % len(order)
                for mode in order[offset:]+order[:offset]:
                    row = measure(modes[mode], data[1][qid], data[2][qid])
                    row.update(query_id=qid, repetition=repetition)
                    observations[mode].append(row)
                if i % 50 == 0:
                    print(name, 'evaluation', repetition+1, i, flush=True)
        result = {}
        for mode, values in observations.items():
            times = sorted(r['ms'] for r in values)
            result[mode] = {key: statistics.fmean(r[key] for r in values)
                           for key in ('recall', 'ndcg', 'ms', 'embedding_calls')}
            result[mode].update(median_ms=statistics.median(times),
                               p95_ms=times[math.ceil(.95*len(times))-1], per_query=values)
        report = dict(dataset=name, documents=len(data[0]), queries=len(query_ids),
                      repetitions=2, top_k=10, policy_sha256=policy_hash,
                      timing='Warm in-process end-to-end search including uncached query tokenization and ONNX embedding; excludes index/model startup',
                      hardware=profile_hardware().to_dict(), provenance=provenance,
                      results=result)
        (args.output/('benchmark-phase9-'+name+'.json')).write_text(json.dumps(report, indent=2))
        print(name, {m: {k:v for k,v in r.items() if k != 'per_query'}
                     for m,r in result.items()}, flush=True)

    if not args.resume_frozen:
        evaluate('cranfield-heldout', cran, test_ids, lexical, fusion)
    url = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
    target = args.work/'scifact.zip'
    if not target.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        # Validate before caching; never extract untrusted archive paths.
        import io
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise ValueError('corrupt dataset zip')
        target.write_bytes(payload)
    if hashlib.md5(target.read_bytes()).hexdigest() != '5f7d1de60b170fc8027bb7898e2efca1':
        raise ValueError('SciFact checksum differs from the BEIR published archive')
    provenance = dict(provenance, scifact=dict(url=url,
                     sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    with zipfile.ZipFile(target) as archive:
        corpus = [json.loads(line) for line in archive.read('scifact/corpus.jsonl').splitlines()]
        queries = {d['_id']: d['text'] for d in
                   map(json.loads, archive.read('scifact/queries.jsonl').splitlines())}
        qrels = {}
        for line in archive.read('scifact/qrels/test.tsv').decode().splitlines()[1:]:
            qid, did, grade = line.split()
            qrels.setdefault(qid, {})[did] = int(grade)
    assert len(corpus) == 5183 and len(qrels) == 300
    data = ([Chunk(d['_id'], d['_id'], d.get('title', '')+' '+d['text'])
             for d in corpus], queries, qrels)
    lexical, fusion = setup('scifact', data[0])
    evaluate('scifact-transfer', data, sorted(qrels), lexical, fusion)


if __name__ == '__main__':
    main()
