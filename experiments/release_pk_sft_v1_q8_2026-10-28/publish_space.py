#!/usr/bin/env python3
"""Upload the staged ember-chat Space folder (pk_sft_v1 F16) to huggingface.co/spaces/johnnyteutonic/ember-chat.
Runs only on Jonathan's go for this release.

    python3 publish_space.py --space-dir DIR --env-file FILE --confirm ember-chat [--dry-run]

- The token: the one line of FILE that starts with HF_TOKEN= is read; no other line is parsed, kept or
  printed, and the token itself is never printed.
- Before uploading: every file in DIR must match the sha256 in DIR/MANIFEST.sha256 (written when the
  folder was staged), and the token's account must be johnnyteutonic.
- One commit to the Space repo (upload_folder): the files in DIR replace the files of the same name;
  nothing else in the Space is deleted. MANIFEST.sha256 is not uploaded.
"""
import argparse
import hashlib
import os
import sys

OWNER, SPACE = "johnnyteutonic", "johnnyteutonic/ember-chat"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hf_token(env_file):
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--space-dir", required=True)
ap.add_argument("--env-file", required=True)
ap.add_argument("--confirm", default="")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
if a.confirm != "ember-chat":
    sys.exit("needs --confirm ember-chat (Jonathan's go for this release)")
man = os.path.join(a.space_dir, "MANIFEST.sha256")
bad = []
listed = set()
for line in open(man):
    h, name = line.split(None, 1)
    name = name.strip()
    listed.add(name)
    if sha256(os.path.join(a.space_dir, name)) != h:
        bad.append(name)
on_disk = {os.path.relpath(os.path.join(r, f), a.space_dir).replace(os.sep, "/")
           for r, _, fs in os.walk(a.space_dir) for f in fs} - {"MANIFEST.sha256"}
if bad or on_disk != listed:
    sys.exit(f"staged folder does not match its manifest: changed {bad}, unlisted {sorted(on_disk - listed)}, "
             f"missing {sorted(listed - on_disk)}")
print(f"{len(listed)} files match MANIFEST.sha256")
token = hf_token(a.env_file)
if not token:
    sys.exit(f"no HF_TOKEN line in {a.env_file}")
from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=token)
who = api.whoami()["name"]
if who.lower() != OWNER:
    sys.exit(f"the token belongs to {who}, not {OWNER}: stopping")
src = open(os.path.join(a.space_dir, "SOURCE.txt")).read().strip().splitlines()
msg = "pk_sft_v1 F16 (97.5M parameters, 221 MB) with ember.cpp " + src[0].split(":", 1)[1].strip()[:10] + \
      ": greedy F16 = F32, WebAssembly = native, WebGPU = WebAssembly 40/40 (experiments/release_pk_sft_v1_q8_2026-10-28)"
print("commit message:", msg)
if a.dry_run:
    print("dry run: nothing uploaded")
    sys.exit(0)
info = api.upload_folder(folder_path=a.space_dir, repo_id=SPACE, repo_type="space", commit_message=msg,
                         ignore_patterns=["MANIFEST.sha256"])
print("published:", getattr(info, "oid", info))
