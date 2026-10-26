"""The corpus draft table of draft_2026-10-27: an n-gram table of the most frequent next token, built offline from
paperkiln's chat text, read by ember's EMBER_DRAFT=table drafter (draft_table.h) and by dr_sim.py.

    python3 dr_table.py tokenize VOCAB.gguf OUT.u16 [--max-bytes N] TEXT...
    python3 dr_table.py build CORPUS.u16[,CORPUS2.u16...] OUT.bin [--orders 4] [--min-count 4] [--min-conf 0.25]
                        [--prune-redundant 1]
    python3 dr_table.py info TABLE.bin

tokenize: each TEXT (paperkiln's chat-format text.txt: <|user|>...<|assistant|>...<|endoftext|>, the format the
  serving template writes) through the model's own vocabulary (paperkiln tools/llama3_bpe.py, the specials parsed
  out first, as ember encodes a prompt), at most --max-bytes of each file (whole lines), as little-endian uint16.
build: for every order o = 1..orders and every context c (the last o tokens) seen at least --min-count times, the
  most frequent next token t (ties: the smaller id) and its share p = count(c, t) / count(c). An entry is kept when
  p >= --min-conf and, with --prune-redundant, when the next-lower order's lookup of the same context would not
  give the same token anyway (a back-off that answers the same is redundant).
Keys (draft_table.h has the same): t0 is the most recent token; for o <= 3 the key is the packing
  t0 | t1 << 16 | t2 << 32; for o = 4 it is mix64(t0 | t1 << 16 | t2 << 32 | t3 << 48) >> 16 (48 bits, splitmix64's
  finaliser; a collision only costs a wrong draft, which verification rejects).
File (little-endian): b"EMBERDT1", u32 vocab_size, u32 orders, u64 entries[orders]; then per order, ascending
  keys u64[n], next u16[n], conf u8[n] (round(255 p)), each array zero-padded to 8 bytes."""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

MAGIC = b"EMBERDT1"
M48 = np.uint64((1 << 48) - 1)


def mix64(x):
    x = x.copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def context_keys(ids, o):
    """Key of the context ending at each position i >= o - 1 (ids[i] the most recent), for positions o-1 .. n-2
    (each has a next token)."""
    n = len(ids)
    ids = ids.astype(np.uint64)
    end = n - 1  # contexts whose next token exists
    k = np.zeros(end - (o - 1), dtype=np.uint64)
    for j in range(o):
        k |= ids[o - 1 - j:end - j] << np.uint64(16 * j)
    if o == 4:
        k = mix64(k) >> np.uint64(16)
    return k


def key_of(ctx):
    """The key of one context (a list, most recent last), as draft_table.h computes it."""
    o = len(ctx)
    k = 0
    for j in range(o):
        k |= int(ctx[-1 - j]) << (16 * j)
    if o == 4:
        k = int(mix64(np.array([k], dtype=np.uint64))[0] >> np.uint64(16))
    return k


def tokenize(a):
    sys.path.insert(0, a.pk_tools)
    from llama3_bpe import from_gguf
    tok = from_gguf(a.vocab)
    vs = tok.get_vocab_size()
    assert vs <= 65535, vs
    total = 0
    with open(a.out, "wb") as f:
        for path in a.text:
            got = 0
            batch = []
            with open(path, encoding="utf-8") as src:
                for line in src:
                    if a.max_bytes and got >= a.max_bytes:
                        break
                    got += len(line.encode("utf-8"))
                    batch.append(line)
                    if len(batch) >= 2000:
                        total += emit(tok, batch, f)
                        batch = []
            if batch:
                total += emit(tok, batch, f)
            print(json.dumps({"file": path, "bytes": got, "tokens_so_far": total}), flush=True)
    print(json.dumps({"out": a.out, "tokens": total, "vocab": vs}))


def emit(tok, lines, f):
    n = 0
    for e in tok.encode_batch(lines, add_special_tokens=False):
        arr = np.array(e.ids, dtype=np.uint16)
        arr.tofile(f)
        n += arr.size
    return n


def build(a):
    ids = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in a.corpus.split(",")])
    vocab = int(a.vocab_size)
    assert int(ids.max()) < vocab, (int(ids.max()), vocab)
    tables = []
    for o in range(1, a.orders + 1):
        ck = context_keys(ids, o)
        nxt = ids[o:].astype(np.uint64)
        comb = (ck << np.uint64(16)) | nxt
        del ck, nxt
        comb.sort()
        vals, counts = np.unique(comb, return_counts=True)
        del comb
        ctx = vals >> np.uint64(16)
        tokn = (vals & np.uint64(0xFFFF)).astype(np.uint16)
        del vals
        # per context: the largest count (ties: the smaller next id, first in this ascending order)
        starts = np.flatnonzero(np.concatenate(([True], ctx[1:] != ctx[:-1])))
        tot = np.add.reduceat(counts, starts)
        order = np.lexsort((-counts.astype(np.int64), ctx))  # ctx ascending, count descending, stable in next
        first = order[starts]  # the same group boundaries: ctx ascending in both
        best_t = tokn[first]
        best_c = counts[first]
        keys = ctx[starts]
        del ctx, tokn, counts, order
        p = best_c / tot
        keep = (tot >= a.min_count) & (p >= a.min_conf)
        tables.append({"keys": keys[keep], "next": best_t[keep], "conf": np.round(p[keep] * 255).astype(np.uint8),
                       "contexts_seen": int(keys.size), "kept_before_prune": int(keep.sum())})
        print(json.dumps({"order": o, "contexts": int(keys.size), "kept": int(keep.sum())}), flush=True)
    if a.prune_redundant:
        # an order-o entry whose context's lower-order lookup (with back-off) gives the same token is dropped
        for o in range(a.orders, 1, -1):
            t = tables[o - 1]
            pred = lower_lookup(tables, o - 1, t["keys"], o)
            keep = pred != t["next"].astype(np.int64)
            for f in ("keys", "next", "conf"):
                t[f] = t[f][keep]
            print(json.dumps({"order": o, "after_prune": int(keep.sum())}), flush=True)
    write(a.out, vocab, tables)
    info_path(a.out)


def lower_lookup(tables, upto, keys_hi, o_hi):
    """For order-o_hi context keys, the back-off prediction from orders upto..1 (-1 when none). Needs the lower
    orders' keys of the same contexts: exact for o_hi <= 3 (a prefix of the packing); for o_hi = 4 the hashed key
    has lost the tokens, so order 4 is compared against order 3 only through stored contexts (see build_ctx)."""
    res = np.full(keys_hi.size, -1, dtype=np.int64)
    if o_hi == 4:
        return res  # hashed: no lower key recoverable; order-4 entries are kept (pruned in build_ctx when known)
    for o in range(upto, 0, -1):
        mask = np.uint64((1 << (16 * o)) - 1)
        lk = keys_hi & mask
        t = tables[o - 1]
        idx = np.searchsorted(t["keys"], lk)
        idx = np.minimum(idx, max(t["keys"].size - 1, 0))
        hit = (t["keys"].size > 0) & (t["keys"][idx] == lk) & (res < 0)
        res[hit] = t["next"][idx[hit]].astype(np.int64)
    return res


def write(path, vocab, tables):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", vocab, len(tables)))
        for t in tables:
            f.write(struct.pack("<Q", t["keys"].size))
        for t in tables:
            for arr, dt in ((t["keys"], "<u8"), (t["next"], "<u2"), (t["conf"], "u1")):
                b = arr.astype(dt).tobytes()
                f.write(b)
                f.write(b"\0" * ((-len(b)) % 8))


def read(path):
    buf = open(path, "rb").read()
    assert buf[:8] == MAGIC, "not a draft table"
    vocab, orders = struct.unpack_from("<II", buf, 8)
    ns = struct.unpack_from("<" + "Q" * orders, buf, 16)
    p = 16 + 8 * orders
    tables = []
    for n in ns:
        t = {}
        for name, dt, sz in (("keys", "<u8", 8), ("next", "<u2", 2), ("conf", "u1", 1)):
            t[name] = np.frombuffer(buf, dtype=dt, count=n, offset=p)
            p += n * sz + ((-(n * sz)) % 8)
        tables.append(t)
    assert p == len(buf), (p, len(buf))
    return vocab, tables


def info_path(path):
    vocab, tables = read(path)
    print(json.dumps({"table": path, "bytes": os.path.getsize(path), "vocab": vocab,
                      "entries": [int(t["keys"].size) for t in tables],
                      "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokenize")
    t.add_argument("vocab")
    t.add_argument("out")
    t.add_argument("text", nargs="+")
    t.add_argument("--max-bytes", type=int, default=0)
    t.add_argument("--pk-tools", default="/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
    b = sub.add_parser("build")
    b.add_argument("corpus")
    b.add_argument("out")
    b.add_argument("--vocab-size", type=int, default=16384)
    b.add_argument("--orders", type=int, default=4)
    b.add_argument("--min-count", type=int, default=4)
    b.add_argument("--min-conf", type=float, default=0.25)
    b.add_argument("--prune-redundant", type=int, default=1)
    i = sub.add_parser("info")
    i.add_argument("table")
    a = ap.parse_args()
    if a.cmd == "tokenize":
        tokenize(a)
    elif a.cmd == "build":
        build(a)
    else:
        info_path(a.table)


if __name__ == "__main__":
    main()
