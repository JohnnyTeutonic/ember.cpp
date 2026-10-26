#!/usr/bin/env python3
"""The revision 1.1 self-test receipt (HEAD's adapter_selftest_cpu.json) against a new run: every check of the old
receipt with its recorded fields, byte for byte (as JSON values), except the "bin" path; lists what differs.
Usage: ir12_cmp_selftest.py OLD.json NEW.json"""
import json
import sys

old, new = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
same, diff, missing = 0, [], []
for k, v in old["checks"].items():
    if k not in new["checks"]:
        missing.append(k)
    elif new["checks"][k] == v:
        same += 1
    else:
        diff.append(k)
added = [k for k in new["checks"] if k not in old["checks"]]
print(f"old checks {len(old['checks'])}: {same} identical, {len(diff)} differ, {len(missing)} missing; "
      f"new checks {len(new['checks'])} ({len(added)} added); requests old {old['requests']} new {new['requests']}")
print("hello equal:", old["hello"] == new["hello"])
for k in diff:
    print("DIFF", k)
    print("   old:", json.dumps(old["checks"][k])[:400])
    print("   new:", json.dumps(new["checks"][k])[:400])
for k in missing:
    print("MISSING", k)
print("added:", added)
print("new failures:", new.get("failures"))
