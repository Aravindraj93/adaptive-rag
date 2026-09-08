import multiprocessing
import tempfile
import unittest
from pathlib import Path
from adaptive_rag import Chunk, BM25Retriever, MMapBM25Retriever
from adaptive_rag import SegmentedBM25Index, SegmentedBM25Retriever


def reader_worker(path, ready):
    with SegmentedBM25Retriever(path):
        ready.set()
        import time
        time.sleep(30)


class CleanupProcessTests(unittest.TestCase):
    def test_other_process_reader_pins_segment_until_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index'
            manager = SegmentedBM25Index.create(path, [Chunk('a', 'd', 'shared text')])
            ready = multiprocessing.Event()
            worker = multiprocessing.Process(target=reader_worker, args=(path, ready))
            worker.start()
            try:
                self.assertTrue(ready.wait(10))
                manager.compact()
                self.assertEqual(manager.cleanup(dry_run=False), [])
            finally:
                worker.terminate()
                worker.join(10)
            self.assertEqual(len(manager.cleanup(dry_run=False)), 1)

    def test_bm25_ties_match_after_disk_roundtrip(self):
        chunks = [Chunk('z', 'd', 'shared text'), Chunk('a', 'd', 'shared text')]
        memory = BM25Retriever()
        memory.add(chunks)
        with tempfile.TemporaryDirectory() as directory:
            with MMapBM25Retriever.build(Path(directory) / 'index', chunks) as disk:
                self.assertEqual([r.chunk.id for r in memory.search('shared')],
                                 [r.chunk.id for r in disk.search('shared')])
