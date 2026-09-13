# Reading the data

Run `python examples/inspect_data.py` from the repository root to validate and summarize both datasets. The example needs only Python's standard library.

For pandas users (`python -m pip install pandas`):

```python
import pandas as pd
from examples.read_data import load_trials

df = pd.DataFrame(load_trials("small"))
print(df[["sub_id", "problem_id", "choice", "word_count"]].head())
print(df.groupby("sub_id")["choice"].mean())  # proportion choosing B
print(df.loc[0, "p1"])                       # already a Python list
```

The loader preserves all rows, including blank transcripts. Choose exclusions for your analysis explicitly. It keeps source participant IDs as strings and does not combine IDs across experiments. `rt` and `word_count` retain their exported strings; cast explicitly if needed. No recording files are required.
