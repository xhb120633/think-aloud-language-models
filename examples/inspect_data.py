"""Verify the released inputs and print an aggregate summary. No dependencies."""
import hashlib
import json
import math
try:
    from .read_data import ROOT, FILES, load_trials
except ImportError:
    from read_data import ROOT, FILES, load_trials

def main():
    expected = {"small": (1368, 72, 1323), "large": (44308, 641, 36967)}
    manifest = json.loads((ROOT / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    for size, filename in FILES.items():
        relative = "data/" + filename
        actual_hash = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual_hash == hashes[relative], f"Checksum mismatch: {relative}"
        records = load_trials(size)
        counts = (len(records), len({r["sub_id"] for r in records}),
                  sum(bool(r["think_aloud"].strip()) for r in records))
        assert counts == expected[size], (size, counts)
        for row in records:
            assert row["choice"] in (0, 1)
            for p, v in (("p1", "v1"), ("p2", "v2")):
                assert isinstance(row[p], list) and isinstance(row[v], list)
                assert len(row[p]) == len(row[v]) > 0
                assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in row[p] + row[v])
                assert all(0 <= x <= 100 for x in row[p])
        print(f"{size}: {counts[0]:,} trials; {counts[1]} participants; "
              f"{counts[2]:,} nonblank transcripts; checksum and schema OK")

if __name__ == "__main__":
    main()
