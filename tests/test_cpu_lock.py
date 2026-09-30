import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from benchmarks.cpu_lock import exclusive_cpu, OWNER_ENV


class CpuLockTests(unittest.TestCase):
    def test_competing_process_waits_and_exception_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lock'
            code=('from benchmarks.cpu_lock import exclusive_cpu; import sys\n'
                  'with exclusive_cpu("child",sys.argv[1]): print("acquired",flush=True)\n')
            env=os.environ.copy();env.pop(OWNER_ENV,None)
            with self.assertRaisesRegex(RuntimeError,'release'):
                with exclusive_cpu('parent',path):
                    child=subprocess.Popen([sys.executable,'-c',code,str(path)],
                        stdout=subprocess.PIPE,text=True,env=env)
                    self.assertIn('Waiting',child.stdout.readline())
                    self.assertIsNone(child.poll())
                    raise RuntimeError('release')
            stdout,_=child.communicate(timeout=5)
            self.assertEqual(child.returncode,0)
            self.assertEqual(stdout.strip(),'acquired')

    def test_synchronous_child_can_share_parent_job_without_deadlock(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lock'
            code=('from benchmarks.cpu_lock import exclusive_cpu; import sys\n'
                  'with exclusive_cpu("nested",sys.argv[1]): print("shared",flush=True)\n')
            with exclusive_cpu('parent',path):
                child=subprocess.run([sys.executable,'-c',code,str(path)],
                    capture_output=True,text=True,timeout=5)
                self.assertEqual(child.returncode,0,child.stderr)
                self.assertEqual(child.stdout.strip(),'shared')
