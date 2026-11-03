"""Cross-process exclusion between E10 timing and checkpoint transfer.

A crashed holder leaves the directory in place: fail closed, never steal a lock.
The admission's absolute deadline bounds recovery; no numerical gate is altered.
"""
import json
from pathlib import Path
import time
import uuid


class Gate:
    def __init__(self, directory, role):
        self.path = Path(directory)
        self.role = role
        self.token = uuid.uuid4().hex
        self.held = False

    def acquire(self, wait_seconds=0):
        deadline = time.monotonic() + wait_seconds
        while True:
            if self.path.is_symlink() or any(p.is_symlink() for p in self.path.parents):
                raise ValueError("redirected timing exclusion gate")
            try:
                self.path.mkdir()
            except FileExistsError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.05)
                continue
            # Any crash between mkdir and owner publication also leaves a locked gate.
            (self.path / "owner.json").write_text(json.dumps({"token": self.token, "role": self.role}))
            self.held = True
            return True

    def release(self):
        if not self.held:
            return
        if self.path.is_symlink() or (self.path / "owner.json").is_symlink():
            raise ValueError("redirected timing gate owner")
        owner = json.loads((self.path / "owner.json").read_text())
        if owner != {"token": self.token, "role": self.role}:
            raise ValueError("timing gate ownership changed")
        (self.path / "owner.json").unlink()
        self.path.rmdir()
        self.held = False
