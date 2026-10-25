#!/usr/bin/env python3
"""Generate unicode_nfc_tables.h, the tables behind unicode_nfc.h (Unicode NFC for the Qwen tokeniser).

    python3 tools/gen_nfc_tables.py unicode_nfc_tables.h            # run from ember.cpp, in WSL
    python3 tools/gen_nfc_tables.py --python-only OUT.h            # Python's tables, no HF filter

Source: Python's unicodedata (its Unicode version is recorded in the header). Three tables:
- kCcc: every code point whose canonical combining class is not 0, with its class;
- kDecomp: every code point's full canonical decomposition (applied recursively), Hangul syllables
  excepted (unicode_nfc.h decomposes those arithmetically);
- kComp: the primary composites, keyed by their two-code-point canonical decomposition: a code point
  whose decomposition has two elements and which NFC leaves alone (so the composition exclusions,
  singletons and non-starter decompositions are left out by construction).

HF filter (the default). HF tokenizers' NFC normaliser (the Rust crate unicode-normalization-alignments)
carries older Unicode tables than Python: combining marks and composites added since are inert there.
Qwen's tokenizer.json normalises with it, and ember's ids must equal HF's, so the generator probes
tokenizers.normalizers.NFC and drops what HF does not know: a mark HF does not reorder gets class 0,
a composite HF does not form loses its decomposition and its composition. The dropped code points are
listed in the header. --python-only skips the filter (no tokenizers needed).

kNfcAsciiImages: the ASCII characters that NFC writes for a non-ASCII code point (canonical
singletons such as U+212A KELVIN SIGN -> "K"). NFC can form an added token's text from other text only
through these, which unicode_nfc.h's caller checks at load.
"""
import argparse
import sys
import unicodedata

HANGUL = range(0xAC00, 0xD7A4)


def cps():
    return [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]


def canon(c):
    d = unicodedata.decomposition(chr(c))
    if not d or d.startswith("<"):
        return None
    return [int(x, 16) for x in d.split()]


def full_decomposition(c):
    d = canon(c)
    if d is None:
        return [c]
    out = []
    for x in d:
        out += full_decomposition(x)
    return out


def hf_filter(ccc, comp):
    """The marks and composites HF's NFC does not know: (marks, composites)."""
    from tokenizers import normalizers
    hf = normalizers.NFC().normalize_str
    firsts = {a for a, _ in comp}
    base = next(c for c in range(0x20, 0x7F) if c not in firsts)  # a starter that composes with nothing
    unknown_marks = []
    for m, c in sorted(ccc.items()):
        if c >= 240:  # only U+0345 (Unicode 1.1)
            continue
        s = chr(base) + chr(0x0345) + chr(m)  # class 240 then class c < 240: NFC swaps them
        if hf(s) != unicodedata.normalize("NFC", s):
            if hf(s) != s:
                sys.exit(f"U+{m:04X}: HF reorders it differently from Python (neither known nor inert)")
            unknown_marks.append(m)
    unknown_comp = []
    for (a, b), c in sorted(comp.items()):
        if hf(chr(a) + chr(b)) != chr(c):
            unknown_comp.append(c)
    return unknown_marks, unknown_comp, f"tokenizers {__import__('tokenizers').__version__}"


def ranges(xs):
    out, xs = [], sorted(xs)
    for x in xs:
        if out and out[-1][1] == x - 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return ", ".join(f"U+{a:04X}" if a == b else f"U+{a:04X}-U+{b:04X}" for a, b in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--python-only", action="store_true")
    args = ap.parse_args()

    ccc = {c: unicodedata.combining(chr(c)) for c in cps() if unicodedata.combining(chr(c))}
    decomp = {}
    for c in cps():
        if c in HANGUL:
            continue
        if canon(c) is not None:
            decomp[c] = full_decomposition(c)
    comp = {}
    for c in cps():
        d = canon(c)
        if c in HANGUL or d is None or len(d) != 2:
            continue
        if unicodedata.normalize("NFC", chr(c)) == chr(c):
            comp[(d[0], d[1])] = c

    dropped_marks, dropped_comp, hf_version = [], [], None
    if not args.python_only:
        dropped_marks, dropped_comp, hf_version = hf_filter(ccc, comp)
        for m in dropped_marks:
            del ccc[m]
        for c in dropped_comp:
            del decomp[c]
            comp = {k: v for k, v in comp.items() if v != c}
        # A decomposition must not reach a dropped code point (Unicode's stability rules make this so).
        for c, d in decomp.items():
            assert not set(d) & set(dropped_comp), hex(c)

    images = sorted({ch for c in cps() if c >= 0x80
                     for ch in unicodedata.normalize("NFC", chr(c)) if ord(ch) < 0x80})
    flat, entries = [], []
    for c in sorted(decomp):
        entries.append((c, len(flat), len(decomp[c])))
        flat += decomp[c]

    L = []
    w = L.append
    w("// Generated by tools/gen_nfc_tables.py: do not edit, regenerate.")
    w(f"// Source: Python {sys.version.split()[0]} unicodedata, Unicode {unicodedata.unidata_version}.")
    if hf_version:
        w(f"// Filtered to HF's NFC ({hf_version}, unicode-normalization-alignments' older tables):")
        w(f"// {len(dropped_marks)} combining marks inert there get class 0 here: "
          f"{ranges(dropped_marks)}.")
        w(f"// {len(dropped_comp)} composites HF does not form lose their decomposition and composition: "
          f"{ranges(dropped_comp)}.")
    else:
        w("// Python's tables unfiltered (--python-only).")
    w("#pragma once")
    w("#include <cstdint>")
    w("")
    w("namespace ember {")
    w("namespace nfc {")
    w("namespace tables {")
    w("")
    w(f'inline constexpr char kUnicodeVersion[] = "{unicodedata.unidata_version}";')
    w(f'inline constexpr bool kHfFiltered = {"true" if hf_version else "false"};')
    esc = "".join(ch if ch not in '"\\' else "\\" + ch for ch in images)
    w(f'inline constexpr char kNfcAsciiImages[] = "{esc}";')
    w("")
    w("// Canonical combining class != 0: code points ascending, with their class.")
    w(f"inline constexpr uint32_t kCccCount = {len(ccc)};")
    items = sorted(ccc.items())
    w("inline constexpr uint32_t kCccCp[kCccCount] = {")
    for i in range(0, len(items), 10):
        w("    " + " ".join(f"0x{c:05X}," for c, _ in items[i:i + 10]))
    w("};")
    w("inline constexpr uint8_t kCccVal[kCccCount] = {")
    for i in range(0, len(items), 20):
        w("    " + " ".join(f"{v}," for _, v in items[i:i + 20]))
    w("};")
    w("")
    w("// Full canonical decompositions (Hangul syllables excepted): code point, offset into")
    w("// kDecompData, length.")
    w(f"inline constexpr uint32_t kDecompCount = {len(entries)};")
    w("inline constexpr uint32_t kDecompCp[kDecompCount] = {")
    for i in range(0, len(entries), 10):
        w("    " + " ".join(f"0x{c:05X}," for c, _, _ in entries[i:i + 10]))
    w("};")
    w("inline constexpr uint16_t kDecompOff[kDecompCount] = {")
    for i in range(0, len(entries), 16):
        w("    " + " ".join(f"{o}," for _, o, _ in entries[i:i + 16]))
    w("};")
    w("inline constexpr uint8_t kDecompLen[kDecompCount] = {")
    for i in range(0, len(entries), 32):
        w("    " + " ".join(f"{n}," for _, _, n in entries[i:i + 32]))
    w("};")
    w(f"inline constexpr uint32_t kDecompDataCount = {len(flat)};")
    w("inline constexpr uint32_t kDecompData[kDecompDataCount] = {")
    for i in range(0, len(flat), 10):
        w("    " + " ".join(f"0x{c:05X}," for c in flat[i:i + 10]))
    w("};")
    w("")
    w("// Primary composites: key (first << 32 | second) ascending, and the composite.")
    pairs = sorted(((a << 32) | b, c) for (a, b), c in comp.items())
    w(f"inline constexpr uint32_t kCompCount = {len(pairs)};")
    w("inline constexpr uint64_t kCompKey[kCompCount] = {")
    for i in range(0, len(pairs), 6):
        w("    " + " ".join(f"0x{k:012X}ull," for k, _ in pairs[i:i + 6]))
    w("};")
    w("inline constexpr uint32_t kCompVal[kCompCount] = {")
    for i in range(0, len(pairs), 10):
        w("    " + " ".join(f"0x{c:05X}," for _, c in pairs[i:i + 10]))
    w("};")
    w("")
    w("}  // namespace tables")
    w("}  // namespace nfc")
    w("}  // namespace ember")
    assert len(flat) < 65536
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print(f"Unicode {unicodedata.unidata_version}: ccc {len(ccc)}, decompositions {len(entries)} "
          f"({len(flat)} code points), composites {len(pairs)}, ASCII images {images!r}; "
          f"HF filter: {hf_version or 'off'} (marks dropped {len(dropped_marks)}, composites dropped "
          f"{len(dropped_comp)})")


if __name__ == "__main__":
    main()
