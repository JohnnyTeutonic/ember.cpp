"""Receipt-validator regressions only; no server, model execution or GPU."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import decode_pick_gate as gate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.binary, self.model, self.prompts = (root / n for n in ("server", "model.gguf", "prompts.jsonl"))
        self.binary.write_bytes(b"fake binary: never executed")
        self.model.write_bytes(b"fake model: never loaded")
        self.prompts.write_text('\n{"user_input":"one"}\n\n{"user_input":"two"}\n')
        self.out = root / "receipts"
        with redirect_stdout(io.StringIO()):
            self.manifest = gate.plan(self.binary, self.model, self.prompts, self.out, 8080)
        for run in self.manifest["runs"]:
            on = "_on" in run["name"]
            records = [dict(i=i, turn=0, reply="same", ids=[4, 5], stop_id=2, prompt_ids=[0, 1],
                            tokens=2, wall_ms=3 if on else 4, first_token_ms=1,
                            passes=0, accepted=0, drafted=0, warm_pass_same=True)
                       for i in range(2)]
            if run["dump"]:
                for i, r in enumerate(records):
                    r.update(rows=2, row0=i * 2)
            path = self.out / (run["name"] + ".jsonl")
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            meta = dict(env=run["env"], once=False, gpu_layers="999",
                        bin=str(self.binary), gguf=str(self.model), prompts=str(self.prompts),
                        health=dict(model=str(self.model), deterministic=run["name"].startswith("det_")),
                        vocab=2 if run["dump"] else None, dump_rows=4 if run["dump"] else None)
            Path(str(path) + ".meta.json").write_text(json.dumps(meta))
            if run["dump"]:
                Path(str(path) + ".rows.f32").write_bytes(struct.pack("8f", *range(8)))

    def change_meta(self, name, change):
        path = self.out / (name + ".jsonl.meta.json")
        meta = json.loads(path.read_text())
        change(meta)
        path.write_text(json.dumps(meta))

    def change_record(self, name, change):
        path = self.out / (name + ".jsonl")
        recs = [json.loads(s) for s in path.read_text().splitlines()]
        change(recs)
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))

    def test_valid_and_blank_prompt_lines(self):
        self.assertEqual(gate.analyze(self.out)["status"], "PASS")
        self.assertEqual(self.manifest["request_keys"], [[0, 0], [1, 0]])

    def test_empty_or_truncated_requests_rejected(self):
        self.change_record("default_on_a", lambda r: r.clear())
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_truncated_dump_rejected(self):
        (self.out / "default_rows_on.jsonl.rows.f32").write_bytes(b"short")
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_changed_logit_bits_rejected(self):
        path = self.out / "default_rows_on.jsonl.rows.f32"
        path.write_bytes(bytes([255]) + path.read_bytes()[1:])
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_baseline_nondeterminism_is_inconclusive(self):
        path = self.out / "default_rows_off_repeat.jsonl.rows.f32"
        path.write_bytes(bytes([255]) + path.read_bytes()[1:])
        result = gate.analyze(self.out)
        self.assertEqual(result["status"], "BASELINE_UNSTABLE")
        self.assertFalse(result["modes"]["default"]["baseline_repeat_bits_equal"])

    def test_missing_first_token_timing_rejected(self):
        self.change_record("default_on_a", lambda r: r[0].update(first_token_ms=None))
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_speculation_rejected(self):
        self.change_record("default_on_a", lambda r: r[0].update(passes=1))
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_compact_path_ids_checked_separately(self):
        self.change_record("default_on_a", lambda r: r[0].update(ids=[4, 6]))
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_timing_dump_contamination_rejected(self):
        self.change_meta("default_on_a", lambda m: m.update(dump_rows=20))
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_server_mode_checked(self):
        self.change_meta("det_on_a", lambda m: m["health"].update(deterministic=False))
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_both_timing_pairs_required(self):
        self.change_record("default_on_b", lambda rs: [r.update(wall_ms=4) for r in rs])
        self.assertEqual(gate.analyze(self.out)["status"], "NO_GAIN")

    def test_frozen_input_hash_checked(self):
        self.model.write_bytes(b"different model")
        with self.assertRaises(ValueError):
            gate.analyze(self.out)

    def test_plan_cannot_overwrite_receipts(self):
        with self.assertRaises(FileExistsError):
            gate.plan(self.binary, self.model, self.prompts, self.out, 8080)

    def test_empty_dump_rejected(self):
        self.change_record("default_rows_on", lambda rs: [r.update(rows=0, row0=0) for r in rs])
        with self.assertRaises(ValueError):
            gate.analyze(self.out)


if __name__ == "__main__":
    unittest.main()
