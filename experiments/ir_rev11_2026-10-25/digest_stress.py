"""ember's digest (adapter method "digest") against paperkiln's reference serialiser (tools/ir_validate.py, run as a
black box) on random binary64 patterns, every decimal exponent, several spellings of each number, random strings and
member names. Usage: python3 digest_stress.py ADAPTER OUT_JSON"""
import json
import math
import random
import struct
import subprocess
import sys

sys.path.insert(0, "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos/tools")
import ir_validate  # noqa: E402

BIN, OUT = sys.argv[1], sys.argv[2]
p = subprocess.Popen([BIN, "--conformance"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                     encoding="utf-8")
n_req = 0


def call(text, prose=True):
    global n_req
    n_req += 1
    p.stdin.write(json.dumps({"id": n_req, "method": "digest", "params": {"text": text, "prose": prose}}) + "\n")
    p.stdin.flush()
    r = json.loads(p.stdout.readline())
    return r["result"]


rng = random.Random(20261025)
nums = []
for _ in range(60000):  # random bit patterns, finite
    while True:
        v = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if math.isfinite(v):
            break
    nums.append(v)
for e in range(-330, 309):  # every decimal exponent, a few mantissas each
    for _ in range(8):
        v = float(f"{rng.uniform(1, 10):.{rng.randint(1, 17)}f}e{e}")
        if math.isfinite(v):
            nums.append(v)
special = [5e-324, -5e-324, 2.2250738585072014e-308, 2.225073858507201e-308, 1.7976931348623157e308, 1e21, 1e-7,
           9.999999999999999e20, 1e20, 123456789012345678901.0, 0.1, 0.2, 0.3, 1 / 3, 2 ** 53, 2 ** 53 - 1, 2 ** -1074,
           4.35, 1e-6, 9.5367431640625e-07, 1e300, 1.5, 100.0, 1e22, 1e23]
nums += special + [-x for x in special]
nums = [x for x in nums if not (x == 0 and math.copysign(1, x) < 0)]
spellings = [repr, lambda v: "%.17e" % v, lambda v: "%.25g" % v, lambda v: "%.20E" % v]

bad = []
checked = 0
B = 500
for spell in spellings:
    for i in range(0, len(nums), B):
        chunk = nums[i:i + B]
        toks = []
        for v in chunk:
            t = spell(v)
            if t in ("inf", "-inf"):
                continue
            if "." not in t and "e" not in t.lower():
                t = t + ".0"  # a float token (integer tokens beyond 2^53 are not I-JSON)
            toks.append(t)
        text = "[" + ",".join(toks) + "]"
        want = ir_validate.canonical(text, True)
        got = call(text)
        checked += len(toks)
        if got.get("canonical") != want:
            for t in toks:
                w = ir_validate.canonical(t, True)
                g = call(t).get("canonical")
                if g != w:
                    bad.append({"token": t, "ember": g, "reference": w})

# strings and member names: random code points (no lone surrogates), every escape-relevant character
pool = [chr(c) for c in range(0, 0x80)] + [chr(c) for c in (0x7F, 0x80, 0xA0, 0xE9, 0x2028, 0x2029, 0xFEFF, 0xFFFF,
                                                         0xE000, 0x10000, 0x1F600, 0x10FFFF, 0x301)]
str_bad = []
for _ in range(400):
    obj = {}
    for _ in range(rng.randint(1, 8)):
        k = "".join(rng.choice(pool) for _ in range(rng.randint(0, 6)))
        obj[k] = "".join(rng.choice(pool) for _ in range(rng.randint(0, 10)))
    for text in (json.dumps(obj, ensure_ascii=True), json.dumps(obj, ensure_ascii=False)):
        for prose in (True, False):
            want = ir_validate.canonical(text, prose)
            got = call(text, prose)
            if got.get("canonical") != want or got.get("sha256") != ir_validate.digest(text, prose):
                str_bad.append({"text": text, "prose": prose})
p.stdin.close()
p.wait()
rep = {"numbers_checked": checked, "number_mismatches": bad[:50], "n_number_mismatches": len(bad),
       "string_objects": 400 * 4, "string_mismatches": str_bad[:20], "n_string_mismatches": len(str_bad),
       "requests": n_req}
json.dump(rep, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, list)}))
sys.exit(0 if not bad and not str_bad else 1)
