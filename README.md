# Rethinking Think-Aloud in the Age of Language Models

**Open data and research code for _Rethinking Think-Aloud in the Age of Language Models: Verbally reported thoughts are predictive of behavior_.**

Hanbo Xie, Hua-Dong Xiong, and Robert C. Wilson

The data pair risky choices with concurrent think-aloud transcripts. Each row contains one participant's choice, the probabilities and outcomes of both options, and the associated transcript. The two experiments contain **45,676 trials** in total.

## Get the data

No account, API key, GPU, or special data format is needed. The two UTF-8 CSV files are tracked directly in Git, without Git LFS.

| Dataset | Participants | Trials | Download |
|---|---:|---:|---|
| Experiment 1 (`small`) | 72 | 1,368 | [CSV, 0.43 MB](https://raw.githubusercontent.com/xhb120633/think-aloud-language-models/main/data/behavioral_text_data.csv) |
| Experiment 2 (`large`) | 641 | 44,308 | [CSV, 9.65 MB](https://raw.githubusercontent.com/xhb120633/think-aloud-language-models/main/data/behavioral_text_data_expanded.csv) |

Use **Save link as** to download an individual CSV, or [download the entire repository as a ZIP](https://github.com/xhb120633/think-aloud-language-models/archive/refs/heads/main.zip).

The release contains anonymized tabular data and transcripts. **Audio recordings are not included.** Numeric column headers are retained for compatibility with the research code; the [data dictionary](data/README.md) explains every column.

## Start here: read and check the data

Requires Python 3.10 or newer, with no additional packages. From the repository folder:

```sh
python examples/inspect_data.py
```

This checks file hashes, row counts, participant counts, choices, and gamble lists, then prints a summary. It does not train models, contact an API, or write into the datasets.

For analysis-ready records with descriptive column names:

```python
from examples.read_data import load_trials

trials = load_trials("small")  # use "large" for Experiment 2
print(len(trials))            # 1368
print(trials[0]["p1"])        # parsed list of probabilities
```

A [pandas example](examples/README.md) is also available.

## Find the research code

| Question / workflow | Code |
|---|---|
| How are trials loaded and prompts constructed? | [Data processing](utils/data_processing.py), [prompt templates](utils/prompt_templates.py), [prompt helpers](utils/prompt_utils.py) |
| Can a model predict the current choice from think-aloud? | [Same-trial prediction and controls](exp1_llm_prediction.py) |
| Can examples from other trials help prediction? | [In-context experiments](exp2_in_context.py) |
| How do behavior-only models perform? | [Neural training](exp1_neural_training.py), [cognitive model definitions](models/cognitive_models.py) |
| How are explicit choice statements identified? | [Extraction](exp_superficial_claims.py), [extraction prompts](utils/superficial_action_extraction.py), [summary analysis](analyze_superficial_claims.py) |
| How similar is model reasoning to human transcripts? | [Embedding generation](embed_cot_data.py), [similarity analysis](analyze_cot_embeddings_similarity.py) |
| Which scripts produce the paper's figures and tables? | [Paper-to-code map and run guide](docs/REPRODUCING.md) |

## Running models and analyses

[The run guide](docs/REPRODUCING.md) distinguishes data-only examples, model generation, and analyses requiring saved outputs. Dependency lists are provided for analysis and model execution. Model runs may need GPU memory, access to model weights, or paid API access.

This is a **core-data and source-code release**. Saved predictions, embeddings, checkpoints, human-rater workbooks, transcription-quality comparisons, and external benchmark datasets are not bundled. Consequently, cloning this repository alone does not reproduce every published figure. The guide identifies those dependencies and known legacy entry-point limitations explicitly. No models or paper analyses were rerun when preparing this release.

## Citation and versions

Code and documentation: [MIT](LICENSE). The two CSV datasets: [CC BY 4.0](data/LICENSE.md).

Please cite the accompanying manuscript and the repository version you used. [CITATION.cff](CITATION.cff) provides authorship and a repository citation; no publication DOI is assigned here. [Versioned releases](https://github.com/xhb120633/think-aloud-language-models/releases) provide downloadable snapshots.

[Release provenance](docs/PROVENANCE.md) and [SHA-256 checksums](FILE_MANIFEST.json) document the files. Questions or reproducibility reports can be submitted through [GitHub Issues](https://github.com/xhb120633/think-aloud-language-models/issues).
