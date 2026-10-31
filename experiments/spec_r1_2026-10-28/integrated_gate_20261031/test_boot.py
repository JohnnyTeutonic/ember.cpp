"""No network or GPU: operational preservation and archive integrity tests."""
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile
import boot
import vm

class BootTests(unittest.TestCase):
    def test_zip_clamps_timestamp_without_changing_raw_bytes_or_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'out';source.mkdir(); p=source/'old.txt';p.write_bytes(b'evidence')
            os.utime(p,(0,0)); before=p.stat().st_mtime_ns
            boot.pack(source,root/'out.zip')
            with zipfile.ZipFile(root/'out.zip') as z:
                self.assertEqual(z.read('old.txt'),b'evidence');self.assertIsNone(z.testzip())
            self.assertEqual(p.stat().st_mtime_ns,before)

    def test_existing_terminal_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'summary.json').write_bytes(b'negative')
            with self.assertRaises(FileExistsError):boot.main(root)
            self.assertEqual((root/'summary.json').read_bytes(),b'negative')
            self.assertEqual(sorted(x.name for x in root.iterdir()),['summary.json'])

    def test_archive_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with tarfile.open(root/'bad.tar','w') as t:
                p=tarfile.TarInfo('../escaped');p.size=1;t.addfile(p,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):boot.extract(root/'bad.tar',root/'source')
            self.assertFalse((root/'escaped').exists())

    def test_archive_symlink_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with tarfile.open(root/'bad.tar','w') as t:
                p=tarfile.TarInfo('link');p.type=tarfile.SYMTYPE;p.linkname='/etc/passwd';t.addfile(p)
            with self.assertRaises(ValueError):boot.extract(root/'bad.tar',root/'source')

    def test_empty_source_manifest_refused_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):vm.run(root,root/'output',{'source_files':{}})
            self.assertFalse((root/'output').exists())

    def test_existing_raw_output_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();(out/'negative.json').write_bytes(b'old')
            manifest={'production_source_commit':vm.check.SOURCE,'source_files':{x:'x' for x in
                ('api.cpp','model.cpp','model.h','verify_kernels.cu','verify_kernels.h','row_pick_partitioned.cuh','decode_pick_policy.h','CMakeLists.txt')}}
            with self.assertRaises(FileExistsError):vm.run(root,out,manifest)
            self.assertEqual((out/'negative.json').read_bytes(),b'old')

if __name__=='__main__':unittest.main()
