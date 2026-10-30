"""Run the staged E9 gate on a remote VM; always return explicit diagnostic receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

FILES = ("ember.tgz", "ember_base.tgz", "inputs.tgz", "colab_kvq_vm.sh", "HEAD")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(outdir, content=Path("/content")):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = content / "out"
    output.mkdir(exist_ok=True)
    execution = {"gate_pass": False, "process_exit_code": None, "complete_marker": False}
    try:
        upload = content / "chat/upload"
        manifest = json.loads((upload / "STAGING.json").read_text())
        if (manifest.get("formats") != ["f16"] or manifest.get("deadline_minutes") != 95
                or manifest.get("phases") != ["g1", "g2", "g3", "tfd_qwen3", "tfd_qwen25"]):
            raise ValueError("campaign scope/deadline differs from staged E9 plan")
        for name in FILES:
            source = upload / name
            expected = manifest["files"][name]
            if source.stat().st_size != expected["bytes"] or sha(source) != expected["sha256"]:
                raise ValueError("staged hash mismatch: " + name)
            shutil.copyfile(source, content / name)
        shutil.copyfile(upload / "STAGING.json", output / "STAGING.json")
        baseline = (content / "HEAD").read_text().strip()
        if baseline != manifest["baseline_ember_commit"]:
            raise ValueError("baseline commit differs from manifest")
        phases = ",".join(manifest["phases"])
        env = dict(os.environ, KVQ_FORMATS="f16", KVQ_BUILD_JOBS="4")
        result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=15s", "95m", "bash",
                                 str(content / "colab_kvq_vm.sh"), baseline, phases], env=env, check=False)
        execution["process_exit_code"] = result.returncode
        execution["complete_marker"] = (content / "GATE_DONE").exists()
        report = output / "gate_summary.json"
        summary = json.loads(report.read_text()) if report.exists() else {"pass": False, "errors": ["missing gate summary"]}
        passed = result.returncode == 0 and execution["complete_marker"] and summary.get("pass") is True
        if not passed:
            summary["pass"] = False
            summary.setdefault("errors", []).append("gate process failed, timed out, or lacks completion marker")
        execution["gate_pass"] = passed
    except (OSError, ValueError, KeyError) as exc:
        summary = {"pass": False, "errors": [str(exc)]}
    (output / "gate_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (output / "execution.json").write_text(json.dumps(execution, indent=1) + "\n")
    for name in ("gate_summary.json", "execution.json"):
        shutil.copyfile(output / name, outdir / name)
    log = content / "gate_vm.log"
    (outdir / "vm.log").write_text(log.read_text(errors="replace") if log.exists() else "VM gate did not start\n")
    with zipfile.ZipFile(outdir / "out.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in output.rglob("*"):
            if source.is_file() and source.suffix != ".bin":
                archive.write(source, str(source.relative_to(output)))
    print(json.dumps(execution))
    # Command completion means diagnostics are home. Only gate_summary.pass is a gate verdict.
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir")
    raise SystemExit(run(parser.parse_args().outdir))
