"""Process-owned reader leases for conservative segment reclamation."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import shutil
import uuid

from .writer_lock import writer_lock
from .persistence import read_manifest, write_manifest


@contextmanager
def lease_guard(path):
    directory = Path(path) / '.readers'
    directory.mkdir(exist_ok=True)
    with writer_lock(directory):
        yield directory


def reader_snapshot(initializer):
    @wraps(initializer)
    def initialize(self, path, **kwargs):
        self._lease = None
        with lease_guard(path) as root:
            try:
                initializer(self, path, **kwargs)
                lease = root / uuid.uuid4().hex
                lease.mkdir()
                lock = writer_lock(lease)
                lock.__enter__()
                try:
                    write_manifest(lease / 'snapshot.json', {
                        'segments': [segment.path.name for segment in self._segments]
                    })
                except Exception:
                    lock.__exit__(None, None, None)
                    raise
                self._lease = lock
            except Exception:
                self.close()
                raise
    return initialize


def pinned_segments(root):
    """Called under lease_guard; return live reader pins and remove stale leases."""
    pinned = set()
    for lease in root.iterdir():
        if not lease.is_dir() or lease.is_symlink():
            continue
        try:
            with writer_lock(lease, timeout=0):
                pass
        except TimeoutError:
            pinned.update(read_manifest(lease / 'snapshot.json')['segments'])
        else:
            shutil.rmtree(lease)
    return pinned
