"""Serialize local benchmark and training jobs in the same checkout."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
LOCK_PATH=Path(tempfile.gettempdir())/('chessengine-cpu-'+hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]+'.lock')
OWNER_ENV='CHESSENGINE_CPU_LOCK_OWNER'


@contextmanager
def exclusive_cpu(label, path=LOCK_PATH):
    # Kernel releases the lock after exceptions or process exit; a leftover
    # metadata file does not leave the workspace permanently locked.
    with Path(path).open('a+') as handle:
        handle.seek(0)
        try: owner=json.loads(handle.read()).get('pid')
        except ValueError: owner=None
        inherited=os.environ.get(OWNER_ENV)
        if inherited and str(owner)==inherited:
            # Synchronous subprocesses belong to the parent's already locked
            # CPU job (not a competing job). The parent awaits their exit.
            try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                yield
                return
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Waiting for exclusive CPU access: {label}',flush=True)
            fcntl.flock(handle,fcntl.LOCK_EX)
        try:
            previous=os.environ.get(OWNER_ENV)
            os.environ[OWNER_ENV]=str(os.getpid())
            handle.seek(0);handle.truncate()
            handle.write(json.dumps({'pid':os.getpid(),'label':label}));handle.flush()
            yield
        finally:
            if previous is None: os.environ.pop(OWNER_ENV,None)
            else: os.environ[OWNER_ENV]=previous
            fcntl.flock(handle,fcntl.LOCK_UN)
