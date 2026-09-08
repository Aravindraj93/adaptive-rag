import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_rag import Chunk, HashingEmbedder, LSHDenseRetriever
from adaptive_rag import SegmentedBM25Index, SegmentedBM25Retriever
from adaptive_rag.writer_lock import writer_lock
from adaptive_rag.persistence import read_manifest


def append_worker(path, identifier):
    SegmentedBM25Index(path).append([Chunk(identifier, 'd', 'shared text')])


def lock_worker(path, ready):
    with writer_lock(path):
        ready.set()
        import time
        time.sleep(30)


class ReliabilityTests(unittest.TestCase):
    def test_concurrent_processes_do_not_lose_appends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index'
            SegmentedBM25Index.create(path, [Chunk('base', 'd', 'shared text')])
            workers = [multiprocessing.Process(target=append_worker, args=(path, str(i))) for i in range(3)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(20)
                self.assertEqual(worker.exitcode, 0)
            with SegmentedBM25Retriever(path) as reader:
                self.assertEqual({c.id for c in reader.iter_chunks()}, {'base', '0', '1', '2'})

    def test_process_exit_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = multiprocessing.Event()
            worker = multiprocessing.Process(target=lock_worker, args=(directory, ready))
            worker.start()
            try:
                self.assertTrue(ready.wait(10))
                with self.assertRaises(TimeoutError):
                    with writer_lock(directory, timeout=0.05):
                        pass
            finally:
                worker.terminate()
                worker.join(10)
            with writer_lock(directory, timeout=1):
                pass

    def test_journal_recovers_failed_publication_and_old_reader_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index'
            manager = SegmentedBM25Index.create(path, [Chunk('base', 'd', 'shared text')])
            from adaptive_rag.retrievers import segmented_bm25 as module
            original = module.write_manifest
            def fail_manifest(target, payload):
                if Path(target).name == 'segments.json':
                    raise OSError('simulated interrupted publication')
                return original(target, payload)
            with patch.object(module, 'write_manifest', side_effect=fail_manifest):
                with self.assertRaises(OSError):
                    manager.append([Chunk('new', 'd', 'shared text')])
            self.assertTrue(manager.recover())
            self.assertFalse(manager.recover())
            with SegmentedBM25Retriever(path) as old:
                manager.compact()
                self.assertEqual(len(list(old.iter_chunks())), 2)
            with SegmentedBM25Retriever(path) as new:
                self.assertEqual(len(list(new.iter_chunks())), 2)

    def test_deleted_head_does_not_hide_live_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index'
            manager = SegmentedBM25Index.create(path, [Chunk(str(i), 'd', 'shared text') for i in range(10)])
            manager.delete([str(i) for i in range(9)])
            with SegmentedBM25Retriever(path) as reader:
                self.assertEqual(reader.search('shared', top_k=1)[0].chunk.id, '9')

    def test_lsh_roundtrip_no_embedding_and_incremental_add(self):
        embedder = HashingEmbedder(dimensions=32)
        index = LSHDenseRetriever(embedder, dimensions=32, seed=19)
        index.add([Chunk('apple', 'd', 'apple orchard')])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lsh.json'
            index.save(path)
            def forbidden(texts):
                raise AssertionError('load must not embed')
            restored = LSHDenseRetriever.load(path, embedder=forbidden)
            restored.embedder = embedder
            self.assertEqual(restored.search('apple orchard'), index.search('apple orchard'))
            restored.add([Chunk('ocean', 'd', 'ocean current')])
            self.assertEqual(restored.search('ocean current')[0].chunk.id, 'ocean')

    def test_failed_lsh_batch_leaves_index_unchanged(self):
        index = LSHDenseRetriever(lambda texts: [[1, 0], [1, 0, 0]], dimensions=2)
        with self.assertRaises(ValueError):
            index.add([Chunk('a', 'd', 'alpha'), Chunk('b', 'd', 'beta')])
        self.assertEqual(index._items, [])
