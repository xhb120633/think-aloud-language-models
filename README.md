# Rethinking Think-Aloud in the Age of Language Models

**Open data, study materials, and research code for _Rethinking Think-Aloud in the Age of Language Models: Verbally reported thoughts are predictive of behavior_.**

Authors (identities withheld for anonymous review)

The data pair risky choices with concurrent think-aloud transcripts. Each row contains one participant's choice, the probabilities and outcomes of both options, and the associated transcript. The two experiments contain **45,676 trials** in total.

## Get the data

No account, API key, GPU, or special data format is needed. The two UTF-8 CSV files are tracked directly in Git, without Git LFS.

| Dataset | Participants | Trials | Download |
|---|---:|---:|---|
| Experiment 1: Prospect Theory Design (`small`) | 72 | 1,368 | [CSV, 0.43 MB](data/behavioral_text_data.csv) |
| Experiment 2: Choices13k (`large`) | 641 | 44,308 | [CSV, 9.65 MB](data/behavioral_text_data_expanded.csv) |

Download the CSVs directly, or use the accompanying anonymous repository ZIP.

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

## Study materials

[Collection materials](materials/README.md) include the recovered Experiment 2 jsPsych framework, instructions, consent/debriefing, stimulus definitions, and written instructions (the instructional video is withheld in this review copy). Experiment 1 has a clearly labeled reconstruction from the public data: **two practice problems and all 17 formal problems, without sampling**. The archive and reconstruction document their different provenance and deployment dependencies. Participant recordings are not included.

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

## Anonymous review copy

This snapshot omits author names, affiliations, contact information, the identifying preprint citation, Git history, and the instructional video (which may identify a speaker). Scientific CSVs and Python analysis source are unchanged. Written instructions and stimulus definitions are retained; identifying fields in consent/debriefing have been masked. Experiment 1 still includes both practice items and all 17 formal items without sampling, with its bundled jsPsych library. Experiment 2 remains a historical source archive with missing deployment dependencies.

The omitted video is represented by a notice in the review task. This is not the complete historical participant experience and must not be used for recruitment. Participant recordings are never included. Third-party library credits and licenses remain intact. Author attribution and the preferred preprint citation remain in the separate public repository; identifying links are intentionally absent here.

Code and documentation: [MIT](LICENSE). Datasets: [CC BY 4.0](data/LICENSE.md). Copyright-holder identities are temporarily masked for review. [SHA-256 checksums](FILE_MANIFEST.json) describe this snapshot.
