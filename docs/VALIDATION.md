# Validation for v1.0.0

Date: 2026-09-13. Local environment: CPython 3.12.4 (Anaconda, Windows).

Passed:

- SHA-256 preservation of all 28 copied research files, including both original CSVs.
- Parsing of every CSV row and all gamble probability/outcome lists; matching list lengths, finite numeric values, valid choice codes, and probability bounds.
- Expected row, participant, and nonblank-transcript counts for both experiments.
- Agreement of the new example reader and the original research loader on choices, gamble arrays, and transcripts for every trial.
- Pandas example and original prompt-construction smoke checks.
- Syntax parsing of all 29 Python files and resolution of local Markdown links.
- Pattern scans of the release for credentials and personal machine paths, plus a direct-identifier scan of the two CSVs. No actionable matches were found. This is a limited screening, not an exhaustive privacy guarantee.

Not validated in this release pass:

- GPU/API generation and numerical reproduction of manuscript results.
- Model CLI execution: the local PyTorch installation failed to load its DLL when checking the neural CLI; Transformers and the OpenAI SDK were not installed in this validation environment. Dependency inventories are supplied, but a full model environment was not reconstructed.
- The legacy cognitive CLI, whose source-level interface problems are identified in the run guide.

The dependency-free data inspector is the tested public entry point.
