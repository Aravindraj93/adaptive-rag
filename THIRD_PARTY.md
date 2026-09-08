# External evaluation assets

The runtime package requires no third-party Python dependencies and does not bundle
pretrained model weights or dataset corpora. Optional evaluation tools use numpy,
onnxruntime and tokenizers, installed separately under their respective licenses.

Evaluation sources:

- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (model card: Apache-2.0).
- [Cranfield](https://ir.dcs.gla.ac.uk/resources/test_collections/cran/).
- [BEIR](https://github.com/beir-cellar/beir) dataset distributions of SciFact,
  NFCorpus and ArguAna; original owners and dataset-specific terms remain applicable.

Reports retain query/document identifiers, metrics, hashes, URLs and settings, not
the external corpora or model files. Source licenses and provenance must be reviewed
for the intended use before redistributing external assets. The library's Apache-2.0
license does not relicense any of those assets. No legal certification is claimed.

The complete library license was obtained from the
[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt).
