# Release provenance

Prepared 2026-09-13 from the authors' local research working tree. Relevant Python source and the two primary CSVs were copied byte-for-byte; this is a snapshot of the working files, not a claim that all source matches a previously tagged paper run. Documentation, the standard-library reading examples, and the release checks were added for public access.

The public Git repository begins with this curated snapshot. It does not include private development history, manuscript drafts, review correspondence, recordings, credentials, machine-specific cluster launchers, or auxiliary participant-level files outside the two selected CSVs.

`SOURCE_MANIFEST.json` identifies the copied inputs. `FILE_MANIFEST.json` hashes all release files except itself and generated validation outputs. The manifests allow download-integrity checks; they are not numerical replication certificates.

Validation checks every CSV row, participant and nonblank counts, list-valued fields, source-byte preservation, Python syntax, and documentation links. A fresh-clone quick-start check is also performed before delivery. See the release notes for execution results. The full GPU/API pipeline and paper figures were not rerun.

Version 1.0.1 clarifies the author-confirmed definition of `rt`: milliseconds from stimulus onset to choice, including think-aloud verbalization. The CSVs and scientific analysis code are unchanged; only documentation, an example-code comment, version metadata, and release checksums were updated.

Version 1.0.2 labels the datasets with the manuscript's experiment numbers and designs: Experiment 1 (Prospect Theory Design; `small`) and Experiment 2 (Choices13k; `large`). Data and analysis code are unchanged.

Version 1.1.0 adds author-provided legacy collection materials and a labeled reconstruction of Experiment 1. The original legacy folder is preserved except for redaction of two participant-credit tokens; source/release hashes are in materials/experiment2_choices13k/SOURCE_MANIFEST.json. Experiment 1 restores hard-coded stimuli from all 1,368 CSV rows, uses both practice items and every one of the 17 formal problems (no sampling), and adapts the shared framework for local storage with bundled jsPsych 7.2.3. This is not a recovered historical deployment. Main data and Python analysis code are unchanged. Stimulus/timeline checks and a browser load to the consent screen passed; microphone recording and model analyses were not run. The teaching video is included as an instructional asset; participant recordings remain excluded.

Experiment 1 now includes its own byte-identical copy of the recovered jsPsych package and instructional demonstration video. Its HTML uses local paths, so the folder can be copied independently of Experiment 2. Stimuli and task logic are unchanged.
