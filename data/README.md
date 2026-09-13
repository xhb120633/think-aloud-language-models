# Data dictionary

Each row is one risky-choice trial. The files are UTF-8 CSVs with a header row, and are unchanged from the source data snapshot. Retain row order when comparing with saved results. The column `0` is an exported index, not a new observation variable.

| Original header | Readable name | Meaning |
|---|---|---|
| `0` | `source_index` | Exported row index |
| `1` | `sub_id` | Anonymized participant key; treat as an identifier, not a measured quantity |
| `2` | `choice` | Recorded choice: `0` = option A / option 1, `1` = option B / option 2; some entries use `0.0` or `1.0` |
| `3` | `p1` | List of probabilities for option A, expressed as percentages |
| `4` | `v1` | List of corresponding outcomes for option A, in the task's monetary units |
| `5` | `p2` | List of probabilities for option B, expressed as percentages |
| `6` | `v2` | List of corresponding outcomes for option B, in the task's monetary units |
| `7` | `problem_id` | Identifier for the decision problem |
| `8` | `rt` | Elapsed time in milliseconds (ms), measured from stimulus onset to the participant's choice, including concurrent think-aloud verbalization |
| `9` | `think_aloud` | Concurrent think-aloud transcript; blank text is retained |
| `10` | `word_count` | Stored transcript word count, not recomputed during release |

`p1[i]` and `v1[i]` form one outcome of option A; the same applies to option B. Parse these list-valued strings with `ast.literal_eval`, not `eval`. Transcription wording and stored measurements are retained as supplied.

**Interpreting `rt`:** timing starts when the stimulus appears and ends when the participant makes a choice. This interval includes viewing the stimulus, thinking aloud, and selecting an option. It is therefore a stimulus-to-choice duration under concurrent verbalization, rather than a conventional reaction-time measure from a task without think-aloud. It is not speech duration alone and does not start at speech onset. Values are in milliseconds; divide by 1,000 to express them in seconds.

## Files and counts

| File | Code label | Rows | Participant keys | Nonblank transcript fields |
|---|---|---:|---:|---:|
| [behavioral_text_data.csv](behavioral_text_data.csv) | `small` | 1,368 | 72 | 1,323 |
| [behavioral_text_data_expanded.csv](behavioral_text_data_expanded.csv) | `large` | 44,308 | 641 | 36,967 |

Participant keys should be interpreted within experiment. The research loader encodes them again; those encoded IDs should not be joined to the raw IDs without a mapping. The supplied example retains the source keys and an experiment label.

Nonblank text counts describe these files only and are not an analysis exclusion rule. The manuscript's Experiment 1 valid-transcription count (1,327) differs from the CSV nonblank count by four; the source/version or definition has not been reconciled. No rows have been added, removed, or silently corrected to force agreement.

## Access and privacy

This public release contains anonymized behavioral data and text transcripts only. Audio recordings and participant identity mappings are not provided. The release inspection checked for obvious email addresses, phone numbers, web links, and self-identification phrases; it is not an assertion of exhaustive automated text anonymization.

See [the reading examples](../examples/README.md) for a descriptive-column interface that leaves the original CSVs unchanged.
