import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_pick_fallback import PROMPTS, compare


def evidence():
    rows = [{"config": i, "prompt": p, "ids": [1, 2], "reply": "fixture"}
            for i in range(4) for p in PROMPTS]
    return {(g, p): copy.deepcopy(rows) for g in (0, 1) for p in (0, 1)}


def test_matching_fallback_inventory():
    assert compare(evidence()) == {"pass": True, "requests_per_leg": 12, "legs": 4}


@pytest.mark.parametrize("fault", ["missing_leg", "missing_request", "duplicate", "changed_ids"])
def test_bad_fallback_evidence(fault):
    records = evidence()
    if fault == "missing_leg":
        del records[(1, 1)]
    elif fault == "missing_request":
        records[(1, 1)].pop()
    elif fault == "duplicate":
        records[(1, 0)][-1] = records[(1, 0)][0]
        records[(1, 1)][-1] = records[(1, 1)][0]
    else:
        records[(1, 1)][0]["ids"] = [3]
    with pytest.raises(ValueError):
        compare(records)
