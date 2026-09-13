# Release provenance

Prepared 2026-09-13 from the authors' local research working tree. Relevant Python source and the two primary CSVs were copied byte-for-byte; this is a snapshot of the working files, not a claim that all source matches a previously tagged paper run. Documentation, the standard-library reading examples, and the release checks were added for public access.

The public Git repository begins with this curated snapshot. It does not include private development history, manuscript drafts, review correspondence, recordings, credentials, machine-specific cluster launchers, or auxiliary participant-level files outside the two selected CSVs.

`SOURCE_MANIFEST.json` identifies the copied inputs. `FILE_MANIFEST.json` hashes all release files except itself and generated validation outputs. The manifests allow download-integrity checks; they are not numerical replication certificates.

Validation checks every CSV row, participant and nonblank counts, list-valued fields, source-byte preservation, Python syntax, and documentation links. A fresh-clone quick-start check is also performed before delivery. See the release notes for execution results. The full GPU/API pipeline and paper figures were not rerun.

Version 1.0.1 clarifies the author-confirmed definition of `rt`: milliseconds from stimulus onset to choice, including think-aloud verbalization. The CSVs and scientific analysis code are unchanged; only documentation, an example-code comment, version metadata, and release checksums were updated.
