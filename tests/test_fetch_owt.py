from pathlib import Path

import pytest

from rz1t.fetch_owt_subset import fetch_subset


def test_fetch_subset_rejects_small_and_overwrite(tmp_path):
    out = tmp_path / "owt.jsonl"
    with pytest.raises(ValueError, match="at least 1000"):
        fetch_subset(out, 10)
    out.write_text("{}\n")
    with pytest.raises(ValueError, match="overwrite"):
        fetch_subset(out, 1000)
