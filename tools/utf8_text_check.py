#!/usr/bin/env python3
"""ember::utf8_valid (utf8_text.h) against Python's bytes.decode("utf-8", "replace") on random byte strings.

    python3 tools/utf8_text_check.py BUILD/utf8_text_test [N]

Each string mixes ASCII, lead bytes, continuation bytes and whole characters, cut at random, as BPE byte
tokens are. Exit 1 on any difference."""
import random
import subprocess
import sys


def sample(rng):
    parts = []
    for _ in range(rng.randint(0, 12)):
        k = rng.random()
        if k < 0.3:
            parts.append(bytes([rng.randint(0, 255)]))
        elif k < 0.5:
            parts.append(bytes([rng.randint(0x80, 0xBF)]))
        else:
            ch = chr(rng.choice([rng.randint(0x20, 0x7E), rng.randint(0x80, 0x7FF), rng.randint(0x800, 0xD7FF),
                                 rng.randint(0xE000, 0xFFFF), rng.randint(0x10000, 0x10FFFF)]))
            b = ch.encode()
            parts.append(b[:rng.randint(1, len(b))] if rng.random() < 0.3 else b)
    return b"".join(parts)


def main():
    exe, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    rng = random.Random(26102026)
    xs = [sample(rng) for _ in range(n)]
    r = subprocess.run([exe, "hex"], input="".join(x.hex() + "\n" for x in xs), capture_output=True, text=True,
                       check=True)
    got = r.stdout.split("\n")[:n]
    bad = [(x.hex(), g) for x, g in zip(xs, got) if g != x.decode("utf-8", "replace").encode().hex()]
    for x, g in bad[:10]:
        print(f"FAIL {x}: ember {g}, python {bytes.fromhex(x).decode('utf-8', 'replace').encode().hex()}")
    print(f"{'FAIL' if bad else 'ok'}: {n - len(bad)} of {n} random byte strings as Python's decode(errors='replace')")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
