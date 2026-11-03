"""drive.launch_code run for real, twice, against a temporary root: a delayed duplicate launch arriving while the
first launch's script is running must not touch that script, and must do no work. python3 -m unittest test_launch_code"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drive import launch_code  # noqa: E402

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def wait_for(path, seconds=20):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


class LaunchCodeTests(unittest.TestCase):
    def test_a_delayed_duplicate_launch_neither_rewrites_nor_duplicates_the_running_script(self):
        root = Path(tempfile.mkdtemp())
        # a stand-in for vm_gemma.sh: the same atomic claim, then work that is still being read when the
        # duplicate arrives, padded so the end of the file is read only after the sleep
        padding = "".join(f"# padding line {i}\n" for i in range(2000))
        script = (f"#!/usr/bin/env bash\nR={root}\n"
                  'mkdir "$R/run_$2" 2>/dev/null || { echo refused > "$R/second.txt"; exit 2; }\n'
                  'echo started > "$R/first.txt"\nsleep 2\n' + padding +
                  'echo complete > "$R/done.txt"\n')
        source = root / "source.sh"
        source.write_text(script)
        code = launch_code(source.as_uri(), COMMIT, "t1", root=str(root))

        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertIn("LAUNCHED", r.stdout, r.stderr)
        self.assertTrue(wait_for(root / "first.txt"))
        published = root / "vm_run_t1.sh"
        inode_first = os.stat(published).st_ino

        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)  # the duplicate
        self.assertIn("LAUNCHED", r.stdout, r.stderr)
        self.assertTrue(wait_for(root / "second.txt"))
        self.assertNotEqual(os.stat(published).st_ino, inode_first)  # replaced, never rewritten in place

        self.assertTrue(wait_for(root / "done.txt"))  # the first script ran to its end
        self.assertEqual((root / "second.txt").read_text().strip(), "refused")
        self.assertEqual(sorted(p.name for p in root.iterdir() if p.name.startswith("run_")), ["run_t1"])
        self.assertEqual([p.name for p in root.glob("vm_run_t1.*.tmp")], [])  # temporary files renamed away


if __name__ == "__main__":
    unittest.main()
