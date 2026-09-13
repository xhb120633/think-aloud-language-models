# Anonymous review notice

The following describes the source archive. In this review copy, identifying fields are masked, source manifests linking back to the public release are omitted, and the teaching video is omitted. See the root README for the exact review-copy scope.

# Experiment 2: Choices13k — archived collection materials

This is the author-provided legacy jsPsych experiment source. The 14,568 gamble definitions in `experiment_setting.json` exactly match the gamble vectors on all 43,026 non-practice rows in the released Experiment 2 CSV. The two practice definitions in `index.html` match its remaining 1,282 rows (641 participants x 2). See `STIMULUS_CHECK.json`.

## Included materials

- `index.html`: experiment logic, participant instructions, two practice trials, consent text, debriefing, and an option to delete recordings containing sensitive information.
- `experiment_setting.json`: Choices13k-based gamble definitions.
- `Think-Aloud-Demo.mp4`: instructional think-aloud demonstration used by the task; this is a teaching asset, not a participant recording.
- `jspsych/`: the supplied JavaScript/CSS files, including the custom audio-button response plugin. Third-party copyright notices are retained.

## Archive status and deployment dependencies

This is a source archive, not a configured runnable deployment. The HTML references `lib/vendors/jspsych-7.1.2/` and `lib/jspsych-7-pavlovia-2022.1.1.js`, which were not present in the recovered folder. The bundled `jspsych/jspsych.js` reports version 7.2.3; it has not been silently substituted for the referenced 7.1.2 runtime. Historical Pavlovia project settings are also not included. A researcher deploying the task must supply appropriate runtime dependencies, configure their own storage and participant-credit integration, and arrange their own ethics approval. Do not use this archive as a live recruitment link.

Two SONA credit tokens were redacted from `index.html` (one was in commented-out code). No task logic, consent text, instructions, randomization, or gamble values were changed. `SOURCE_MANIFEST.json` records original and released hashes without exposing credentials or local usernames. All other included files are byte-for-byte copies. No participant audio or participant records are contained in this folder.

The observed implementation samples a random number of items using its original loop and randomizes presentation order. We have preserved that code as recovered; matching stimuli does not establish an exact historical deployment revision or validate timing, randomization, recording, or upload behavior.
