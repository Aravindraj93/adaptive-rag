"""Full Cranfield evaluation using pinned local ONNX MiniLM assets.

Run with --assets pointing to cran.tar.gz, model.onnx, tokenizer.json and
provenance.json downloaded from the URLs recorded in the provenance file.
Dependencies (evaluation only): numpy, onnxruntime, tokenizers.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import tarfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from adaptive_rag import BM25Retriever, DenseRetriever, Chunk
from adaptive_rag import ReciprocalRankFusionRetriever, EscalatingRetriever
from adaptive_rag.hardware import profile_hardware


def records(text):
    for record in re.split(r'(?m)^\.I ', text)[1:]:
        lines = record.splitlines()
        fields = {'id': lines[0].strip()}
        current = None
        for line in lines[1:]:
            if re.fullmatch(r'\.[A-Z]', line.strip()):
                current = line.strip()[1:]
                fields[current] = []
            elif current:
                fields[current].append(line.strip())
        yield {key: ' '.join(value).strip() if isinstance(value, list) else value
               for key, value in fields.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((args.assets / 'provenance.json').read_text())
    for name, details in provenance['files'].items():
        digest = hashlib.sha256((args.assets / name).read_bytes()).hexdigest()
        if digest != details['sha256']:
            raise ValueError(f'asset checksum mismatch: {name}')
    with tarfile.open(args.assets / 'cran.tar.gz', 'r:*') as archive:
        def read(name):
            return archive.extractfile(name).read().decode('utf-8').replace('\r', '')
        corpus = list(records(read('cran.all.1400')))
        queries = list(records(read('cran.qry')))
        qrels = {}
        for line in read('cranqrel').splitlines():
            qid, did, grade = line.split()
            qrels.setdefault(int(qid), {})[did] = max(0, int(grade))
    assert len(corpus) == 1400 and len(queries) == 225
    chunks = [Chunk(doc['id'], doc['id'], doc.get('T', '') + ' ' + doc.get('W', ''))
              for doc in corpus]
    texts = [chunk.text for chunk in chunks] + [query['W'] for query in queries]
    tokenizer = Tokenizer.from_file(str(args.assets / 'tokenizer.json'))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.assets / 'model.onnx'), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    embeddings = {}
    started = time.perf_counter()
    for start in range(0, len(texts), 16):
        batch = texts[start:start+16]
        encoded = tokenizer.encode_batch(batch)
        arrays = {'input_ids': np.array([e.ids for e in encoded], dtype=np.int64),
                  'attention_mask': np.array([e.attention_mask for e in encoded], dtype=np.int64),
                  'token_type_ids': np.array([e.type_ids for e in encoded], dtype=np.int64)}
        output = session.run(None, {i.name: arrays[i.name] for i in session.get_inputs()})[0]
        mask = arrays['attention_mask'][..., None]
        pooled = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        embeddings.update(zip(batch, pooled.tolist()))
        if start % 160 == 0:
            print(f'Embedded {min(start+16, len(texts))}/{len(texts)}', flush=True)
    embedding_seconds = time.perf_counter() - started
    lexical = BM25Retriever()
    lexical.add(chunks)
    dense = DenseRetriever(lambda batch: [embeddings[text] for text in batch])
    dense.add(chunks)
    fusion = ReciprocalRankFusionRetriever([lexical, dense])
    routed = EscalatingRetriever(lexical, fusion)
    modes = {'bm25': lexical, 'minilm_dense': dense, 'rrf': fusion, 'routed_default': routed}
    results = {}
    for name, retriever in modes.items():
        observations = []
        fallback = 0
        for qid, query in enumerate(queries, 1):
            # Cranfield qrels use sequential query ordinals, not raw .I values.
            grades = qrels[qid]
            relevant = {did for did, grade in grades.items() if grade > 0}
            started = time.perf_counter()
            ranked = retriever.search(query['W'], top_k=10)
            elapsed = (time.perf_counter() - started) * 1000
            ids = [result.chunk.id for result in ranked]
            recall = len(set(ids) & relevant) / len(relevant)
            rr = next((1 / rank for rank, did in enumerate(ids, 1) if did in relevant), 0)
            dcg = sum((2**grades.get(did, 0)-1) / math.log2(rank+1)
                      for rank, did in enumerate(ids, 1))
            ideal = sum((2**grade-1) / math.log2(rank+1)
                        for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:10], 1))
            observations.append(dict(query_id=qid, recall_at_10=recall, mrr_at_10=rr,
                                     ndcg_at_10=dcg / ideal if ideal else 0,
                                     retrieval_ms=elapsed, result_ids=ids))
            if name == 'routed_default':
                fallback += routed.last_route.route == 'fallback'
        results[name] = {
            metric: statistics.fmean(row[metric] for row in observations)
            for metric in ('recall_at_10', 'mrr_at_10', 'ndcg_at_10', 'retrieval_ms')
        }
        results[name].update(fallback_queries=fallback, per_query=observations)
        print(name, {k: v for k, v in results[name].items() if k != 'per_query'}, flush=True)
    report = dict(dataset='Cranfield', documents=1400, queries=225, top_k=10,
                  protocol='Full corpus, title + abstract, positive relevance grades, no tuning',
                  timing='Retrieval uses cached query embeddings; not end-to-end model latency',
                  model_max_tokens=256, model_threads=4, embedding_seconds=embedding_seconds,
                  hardware=profile_hardware().to_dict(), provenance=provenance,
                  packages={p: importlib.metadata.version(p) for p in ('numpy', 'onnxruntime', 'tokenizers')},
                  results=results)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
