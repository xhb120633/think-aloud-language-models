# Experiment 1: Prospect Theory Design — reconstructed materials

This is a reconstruction authorized by the authors, using the released Experiment 1 CSV and the recovered legacy collection framework. It is not the original deployment file. The authors confirmed that Experiment 1 hard-coded the stimulus set and presented every formal problem, without sampling.

`stimuli.js` hard-codes all **17 formal problems**. `stimuli.json` also lists the **two practice problems**. Their problem IDs, option sides, probabilities and outcomes are copied exactly from the released CSV; all 72 participants have the same 19 definitions. The existing practice block is retained. `main_expt` uses all 17 entries once, with no sampling. Its presentation order remains randomized as in the recovered framework; historical ordering was not independently recovered.

`index.html` adapts the Experiment 2 framework, reusing its instructions, consent/debrief text, and recording interface. Dependencies point to the jsPsych 7.2.3 files included in this directory, not the missing historical 7.1.2 installation. Pavlovia and SONA integrations are removed. Session data are downloaded locally at completion; no research data are uploaded. These deployment changes and the reconstructed origin distinguish this file from an archival copy.

## Local preview

This directory contains its own `jspsych/` package and `Think-Aloud-Demo.mp4`; it can be copied independently of Experiment 2. From this directory, run `python -m http.server 8000 --bind 127.0.0.1`, then open http://127.0.0.1:8000/ . Microphone access is needed for the recording portions. The instruction video is a study demonstration, not a participant recording. Do not deploy this reconstruction for recruitment with the historical consent text; a new deployment requires its own approved consent and storage arrangements.

Validation checks every released Experiment 1 row against the recovered definitions, confirms all 17 formal items are passed to the timeline without sampling, and checks JavaScript syntax and local asset references. It does not validate recording or establish numerical reproduction of paper analyses.
