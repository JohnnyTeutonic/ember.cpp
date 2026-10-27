#!/usr/bin/env python3
"""ember::utf8_valid (utf8_text.h) against Python's bytes.decode("utf-8", "replace") on random byte strings, and
ember::ReplyStream (the streamed reply) against a reference built on Python's decoder.

    python3 tools/utf8_text_check.py BUILD/utf8_text_test [N]

Each string mixes ASCII, lead bytes, continuation bytes and whole characters, cut at random, as BPE byte
tokens are. The streamed strings add whitespace at the ends and inside, and split at random token boundaries
(0 to 4 bytes a token). After every token the text ember has sent must equal the reference: the bytes so far
less a cut character at the end (the trailing bytes, at most 3, that some continuation completes into a valid
character: a search over continuation bytes at the edges of every Table 3-7 range, each candidate checked by
Python's strict decoder), decoded with errors="replace"; leading whitespace skipped, trailing whitespace held
unless a cut character follows it. After the end: the plain reply (the bytes stripped, then decoded). So a
streamed piece never replaces bytes a later token completes, holds back nothing that is already final, and
the pieces join to the non-streamed reply byte for byte. (CPython's incremental decoder is no reference for
the second point: near the end of its input it waits on bytes that can never become valid, such as ED A0.)
Exit 1 on any difference."""
import functools
import itertools
import random
import subprocess
import sys

WS = " \t\n\v\f\r"


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


def ws(rng):
    return "".join(rng.choice(WS) for _ in range(rng.choice([0, 0, 1, 2, 3]))).encode()


def stream_sample(rng):
    x = ws(rng) + b"".join(sample(rng) + ws(rng) for _ in range(rng.randint(1, 3)))
    cuts, at = [], 0
    while at < len(x):
        at = min(len(x), at + rng.choice([0, 1, 1, 1, 2, 2, 3, 4]))
        cuts.append(at)
    return x, cuts


EDGES = [0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF]  # the ends of every second-byte range in Table 3-7


@functools.lru_cache(maxsize=None)
def completable(tail):
    """True when some continuation bytes complete tail into one valid character (tail its proper prefix)."""
    if not 0xC2 <= tail[0] <= 0xF4 or any(not 0x80 <= b <= 0xBF for b in tail[1:]):
        return False  # only a lead byte and continuation bytes can start one character (a quick reject)
    for total in range(len(tail) + 1, 5):
        for rest in itertools.product(EDGES, repeat=total - len(tail)):
            try:
                if len((tail + bytes(rest)).decode("utf-8")) == 1:
                    return True
            except UnicodeDecodeError:
                pass
    return False


def reference_stream(x, cuts):
    """The text a stream must have sent after each token, then after the end."""
    totals = []
    for c in cuts:
        p = x[:c]
        cut = next((t for t in (3, 2, 1) if len(p) >= t and completable(p[-t:])), 0)
        sent = p[:len(p) - cut].decode("utf-8", "replace").lstrip(WS)
        totals.append(sent if cut and sent else sent.rstrip(WS))
    totals.append(x.decode("utf-8", "replace").strip(WS))
    return totals


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

    srng = random.Random(27102026)
    ss = [stream_sample(srng) for _ in range(n)]
    r = subprocess.run([exe, "stream"], input="".join(f"{x.hex()} {','.join(map(str, c))}\n" for x, c in ss),
                       capture_output=True, text=True, check=True)
    sgot = r.stdout.split("\n")[:n]
    sbad = []
    for (x, cuts), g in zip(ss, sgot):
        want = reference_stream(x, cuts)
        plain = x.strip(WS.encode()).decode("utf-8", "replace")
        want_hex = ",".join(t.encode().hex() for t in want)
        if g != want_hex or want[-1] != plain:
            sbad.append((x.hex(), cuts, g, want_hex))
    for x, cuts, g, w in sbad[:10]:
        print(f"FAIL stream {x} cuts {cuts}: ember {g}, reference {w}")
    print(f"{'FAIL' if sbad else 'ok'}: {n - len(sbad)} of {n} random streamed replies as the reference after "
          f"every token, joined equal to the plain reply")
    sys.exit(1 if bad or sbad else 0)


if __name__ == "__main__":
    main()
