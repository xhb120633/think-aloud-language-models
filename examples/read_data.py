"""Read original CSVs into named records without changing their contents."""
import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {"small": "behavioral_text_data.csv", "large": "behavioral_text_data_expanded.csv"}
COLUMNS = ["source_index", "sub_id", "choice", "p1", "v1", "p2", "v2",
           "problem_id", "rt", "think_aloud", "word_count"]

def load_trials(experiment="small"):
    """Return all rows; preserve blank text and source participant identifiers."""
    if experiment not in FILES:
        raise ValueError("experiment must be 'small' or 'large'")
    records = []
    with (ROOT / "data" / FILES[experiment]).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [str(i) for i in range(11)]:
            raise ValueError("Unexpected CSV schema")
        for row in reader:
            record = {name: row[str(i)] for i, name in enumerate(COLUMNS)}
            record["experiment"] = experiment
            record["choice"] = int(float(record["choice"]))
            for name in ("p1", "v1", "p2", "v2"):
                record[name] = ast.literal_eval(record[name])
            # Preserve response-time and stored word-count strings; no assumed units.
            records.append(record)
    return records
