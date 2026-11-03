"""apt_reconcile.py on fake /etc/apt trees. python3 -m unittest test_apt_reconcile -v"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import apt_reconcile as ar

URI = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
KEY = "/usr/share/keyrings/vm-gemma-cuda-archive-keyring.gpg"


class AptReconcileTests(unittest.TestCase):
    def tree(self, sources_list="", files=None):
        root = Path(tempfile.mkdtemp())
        etc = root / "apt"
        (etc / "sources.list.d").mkdir(parents=True)
        (etc / "sources.list").write_text(sources_list)
        for name, text in (files or {}).items():
            (etc / "sources.list.d" / name).write_text(text)
        return root, etc

    def run_it(self, root, etc):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return ar.main(["--apt-etc", str(etc), "--uri", URI, "--keyring", KEY,
                            "--entry", str(etc / "sources.list.d" / "vm-gemma-cuda-12-6.list"),
                            "--backup", str(root / "backup")])

    def test_the_package_owned_list_already_commented_out_gets_our_active_entry(self):
        # the round-6 case: cuda-keyring installed, its own list commented out; nothing active for the repository
        own = f"# deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] {URI}/ /\n"
        root, etc = self.tree(files={"cuda-ubuntu2404-x86_64.list": own})
        self.assertEqual(self.run_it(root, etc), 0)
        self.assertEqual((etc / "sources.list.d" / "cuda-ubuntu2404-x86_64.list").read_text(), own)
        self.assertEqual((etc / "sources.list.d" / "vm-gemma-cuda-12-6.list").read_text(),
                         f"deb [signed-by={KEY}] {URI}/ /\n")
        self.assertEqual(ar.active_entries(etc, URI), [str(etc / "sources.list.d" / "vm-gemma-cuda-12-6.list")])

    def test_conflicting_one_line_entries_are_commented_with_and_without_slash(self):
        text = (f"deb [arch=amd64 signed-by=/other.gpg] {URI}/ /\n"
                f"deb {URI} /\n"
                f"deb {URI}/extra /\n"                       # a longer path: another repository
                f"deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /\n"
                "deb http://archive.ubuntu.com/ubuntu noble main\n")
        root, etc = self.tree(files={"cuda-old.list": text})
        self.assertEqual(self.run_it(root, etc), 0)
        lines = (etc / "sources.list.d" / "cuda-old.list").read_text().splitlines()
        self.assertTrue(lines[0].startswith(ar.MARK + "deb [arch=amd64"))
        self.assertTrue(lines[1].startswith(ar.MARK + "deb "))
        self.assertEqual(lines[2:], text.splitlines()[2:])
        self.assertEqual((root / "backup" / "cuda-old.list").read_text(), text)

    def test_entries_in_sources_list_itself(self):
        text = f"deb http://archive.ubuntu.com/ubuntu noble main\ndeb [signed-by=/x.gpg] {URI}/ /\n"
        root, etc = self.tree(sources_list=text)
        self.assertEqual(self.run_it(root, etc), 0)
        lines = (etc / "sources.list").read_text().splitlines()
        self.assertEqual(lines[0], "deb http://archive.ubuntu.com/ubuntu noble main")
        self.assertTrue(lines[1].startswith(ar.MARK))

    def test_deb822_only_the_matching_stanza_is_disabled(self):
        text = ("Types: deb\nURIs: http://archive.ubuntu.com/ubuntu\nSuites: noble\nComponents: main\n\n"
                f"# NVIDIA\nTypes: deb\nURIs: {URI}/\nSuites: /\nSigned-By: /x.gpg")
        root, etc = self.tree(files={"mixed.sources": text})
        self.assertEqual(self.run_it(root, etc), 0)
        new = (etc / "sources.list.d" / "mixed.sources").read_text()
        first, second = new.split("\n\n")
        self.assertEqual(first + "\n", text.split("\n\n")[0] + "\n")  # the Ubuntu stanza untouched
        self.assertIn("Enabled: no", second)
        self.assertNotIn("Enabled", first)

    def test_deb822_stanza_with_other_uris_keeps_them(self):
        text = f"Types: deb\nURIs: https://mirror.example/x {URI}\nSuites: /\nEnabled: yes\n"
        root, etc = self.tree(files={"multi.sources": text})
        self.assertEqual(self.run_it(root, etc), 0)
        new = (etc / "sources.list.d" / "multi.sources").read_text()
        self.assertIn("URIs: https://mirror.example/x\n", new)
        self.assertNotIn(URI, new)
        self.assertIn("Enabled: yes", new)

    def test_a_second_run_changes_nothing_further(self):
        root, etc = self.tree(files={"cuda-old.list": f"deb {URI}/ /\n"})
        self.assertEqual(self.run_it(root, etc), 0)
        before = {p.name: p.read_text() for p in (etc / "sources.list.d").iterdir()}
        self.assertEqual(self.run_it(root, etc), 0)
        after = {p.name: p.read_text() for p in (etc / "sources.list.d").iterdir()}
        self.assertEqual(before, after)

    def test_disabled_deb822_stanza_is_not_counted_active(self):
        text = f"Types: deb\nURIs: {URI}/\nSuites: /\nEnabled: no\n"
        root, etc = self.tree(files={"off.sources": text})
        self.assertEqual(self.run_it(root, etc), 0)
        self.assertEqual((etc / "sources.list.d" / "off.sources").read_text(), text)


if __name__ == "__main__":
    unittest.main()
