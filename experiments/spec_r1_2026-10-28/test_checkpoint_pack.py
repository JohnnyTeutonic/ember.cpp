"""Exercise the exact future VM pack code with reproducible source mtimes."""
from pathlib import Path
import os
import zipfile


def test_pack_clamps_archive_metadata_without_mutating_raw_files(tmp_path):
    vm = Path(__file__).with_name("self_output_full_v3_vm.sh").read_text(encoding="utf-8")
    code = vm.split("<<'PACK'\n", 1)[1].split("\nPACK", 1)[0]
    code = code.replace("pathlib.Path('/content')", "pathlib.Path(ROOT)")
    output = tmp_path / "out"
    output.mkdir()
    old = output / "source.py"
    old.write_bytes(b"frozen source\n")
    os.utime(old, ns=(123456789, 0))
    recent = output / "outputs.jsonl"
    recent.write_bytes(b'{"request":1}\n')
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    exec(compile(code, str(Path(__file__)), "exec"), {"ROOT": str(tmp_path)})
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    with zipfile.ZipFile(tmp_path / "out.zip") as archive:
        assert archive.testzip() is None
        assert archive.read("out/source.py") == before["source.py"][0]
        assert archive.getinfo("out/source.py").date_time == (1980, 1, 1, 0, 0, 0)
        assert archive.read("out/outputs.jsonl") == before["outputs.jsonl"][0]
    assert not list(tmp_path.glob("self_tmp_*.zip"))
