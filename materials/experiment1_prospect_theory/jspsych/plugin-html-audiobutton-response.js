var jsPsychHtmlAudioButtonResponse = (function (jspsych) {
    'use strict';
  
    const info = {
        name: "html-audiobutton-response",
        parameters: {
            /** The HTML string to be displayed */
            stimulus: {
                type: jspsych.ParameterType.HTML_STRING,
                default: undefined,
            },
            /** How long to show the stimulus. */
            stimulus_duration: {
                type: jspsych.ParameterType.INT,
                default: null,
            },
            /** How long to show the trial. */
            recording_duration: {
                type: jspsych.ParameterType.INT,
                default: 2000,
            },
            // show_done_button: {
            //     type: jspsych.ParameterType.BOOL,
            //     default: true,
            // },
            // done_button_label: {
            //     type: jspsych.ParameterType.STRING,
            //     default: "Continue",
            // },
            // record_again_button_label: {
            //     type: jspsych.ParameterType.STRING,
            //     default: "Record again",
            // },
            // accept_button_label: {
            //     type: jspsych.ParameterType.STRING,
            //     default: "Continue",
            // },
            // allow_playback: {
            //     type: jspsych.ParameterType.BOOL,
            //     default: false,
            // },
            save_audio_url: {
                type: jspsych.ParameterType.BOOL,
                default: false,
            },
            /** The vertical margin of the button. */
            margin_vertical: {
                type: jspsych.ParameterType.STRING,
                pretty_name: "Margin vertical",
                default: "8px",
            },
            /** The horizontal margin of the button. */
            margin_horizontal: {
                type: jspsych.ParameterType.STRING,
                pretty_name: "Margin horizontal",
                default: "8px",
            },
            /** Array containing the label(s) for the button(s). */
            choices: {
                type: jspsych.ParameterType.STRING,
                pretty_name: "Choices",
                default: undefined,
                array: true,
            },
            /** The HTML for creating button. Can create own style. Use the "%choice%" string to indicate where the label from the choices parameter should be inserted. */
            button_html: {
                type: jspsych.ParameterType.HTML_STRING,
                pretty_name: "Button HTML",
                default: '<button class="jspsych-btn">%choice%</button>',
                array: true,
            },
        },
    };
    /**
     * html-audiobutton-response
     * jsPsych plugin for displaying a stimulus and recording an audio response through a microphone
     * @author Bob Wilson - adapted from html-audio-response and html-button-response by Josh de Leeuw
     */
    class HtmlAudioButtonResponsePlugin {
        constructor(jsPsych) {
        this.jsPsych = jsPsych;
        this.rt = null;
        this.recorded_data_chunks = [];
        this.isResponseRecorded = false;  // initialize it as false
        
            
        }
        trial(display_element, trial) {
            this.recorder = this.jsPsych.pluginAPI.getMicrophoneRecorder();
            this.setupRecordingEvents(display_element, trial);
            this.startRecording();
        }
        
        showDisplay(display_element, trial) {
            const ro = new ResizeObserver((entries, observer) => {
                this.stimulus_start_time = performance.now();
                observer.unobserve(display_element);
                //observer.disconnect();
            });
            ro.observe(display_element);
            let html = `<div id="jspsych-html-audiobutton-response-stimulus">${trial.stimulus}</div>`;
            
            // BOB EDIT
            //display buttons
            var buttons = [];
            if (Array.isArray(trial.button_html)) {
                if (trial.button_html.length == trial.choices.length) {
                    buttons = trial.button_html;
                }
                else {
                    console.error("Error in html-button-response plugin. The length of the button_html array does not equal the length of the choices array");
                }
            }
            else {
                for (var i = 0; i < trial.choices.length; i++) {
                    buttons.push(trial.button_html);
                }
            }

            // if (trial.show_done_button) {
                
                
            html += '<div id="jspsych-html-button-response-btngroup">';
            for (var i = 0; i < trial.choices.length; i++) {
                var str = buttons[i].replace(/%choice%/g, trial.choices[i]);
                html += '<div class="jspsych-html-button-response-button" style="display: inline-block; margin:' +
                trial.margin_vertical +
                " " +
                trial.margin_horizontal +
                '" id="jspsych-html-button-response-button-' +
                i +
                '" data-choice="' +
                i +
                '">' +
                str +
                "</div>";
            }
            html += "</div>";
            // }
            // END BOB EDIT
            display_element.innerHTML = html;
        }
        hideStimulus(display_element) {
            const el = display_element.querySelector("#jspsych-html-audiobutton-response-stimulus");
            if (el) {
                el.style.visibility = "hidden";
            }
        }
        
        addButtonEvent(display_element, trial) {
        for (var i = 0; i < trial.choices.length; i++) {
            let btn = display_element.querySelector("#jspsych-html-button-response-button-" + i);

            btn.addEventListener("click", (e) => {

                if (this.isResponseRecorded) {  // if response is already recorded, do nothing
                    return;
                }
                
                this.button = parseInt(choice);
                btn.classList.add('highlighted');  // add the 'highlighted' class to the clicked button
                
                this.isResponseRecorded = true;  // set it as true once a button is clicked

                const end_time = performance.now();
                this.rt = Math.round(end_time - this.stimulus_start_time);

                var btn_el = e.currentTarget;
                var choice = btn_el.getAttribute("data-choice");

                // Highlight selected button
                btn_el.style.backgroundColor = "";
                btn_el.style.color = "";


                this.button = parseInt(choice);
                // Display 'continue' stimulus instead of immediately stopping the recording
                this.displayContinueButton(display_element, trial);
            });
        }
    }
    
        displayContinueButton(display_element, trial) {
        // Add HTML for continue button to display
        var continueButton = document.createElement("button");
        continueButton.id = "continue-button";
        continueButton.innerText = "Continue";
        continueButton.style.fontSize = "20px"; // Make the font of the button bigger
        continueButton.style.padding = "10px"; // Make the button bigger
        continueButton.style.marginTop = "50px"; // Move the button a bit down

        display_element.appendChild(continueButton);

        // Add event listener for continue button
        continueButton.addEventListener("click", () => {
            this.stopRecording().then(() => {
                this.endTrial(display_element, trial);
            });
        });
    }
        
        setupRecordingEvents(display_element, trial) {
            this.data_available_handler = (e) => {
                if (e.data.size > 0) {
                    this.recorded_data_chunks.push(e.data);
                }
            };
            this.stop_event_handler = () => {
                const data = new Blob(this.recorded_data_chunks, { type: "audio/webm" });
                this.audio_url = URL.createObjectURL(data);
                const reader = new FileReader();
                reader.addEventListener("load", () => {
                    const base64 = reader.result.split(",")[1];
                    this.response = base64;
                    this.load_resolver();
                });
                reader.readAsDataURL(data);
            };
            this.start_event_handler = (e) => {
                // resets the recorded data
                this.recorded_data_chunks.length = 0;
                this.recorder_start_time = e.timeStamp;
                this.showDisplay(display_element, trial);
                this.addButtonEvent(display_element, trial);
                // setup timer for hiding the stimulus
                if (trial.stimulus_duration !== null) {
                    this.jsPsych.pluginAPI.setTimeout(() => {
                        this.hideStimulus(display_element);
                    }, trial.stimulus_duration);
                }
                // setup timer for ending the trial
                if (trial.recording_duration !== null) {
                    this.jsPsych.pluginAPI.setTimeout(() => {
                        // this check is necessary for cases where the
                        // done_button is clicked before the timer expires
                        if (this.recorder.state !== "inactive") {
                            this.stopRecording().then(() => {
                                // if (trial.allow_playback) {
                                //     this.showPlaybackControls(display_element, trial);
                                // }
                                // else {
                                this.endTrial(display_element, trial);
                                // }
                            });
                        }
                    }, trial.recording_duration);
                }
            };
            this.recorder.addEventListener("dataavailable", this.data_available_handler);
            this.recorder.addEventListener("stop", this.stop_event_handler);
            this.recorder.addEventListener("start", this.start_event_handler);
        }
        startRecording() {
            this.recorder.start();
        }
        stopRecording() {
            this.recorder.stop();
            return new Promise((resolve) => {
                this.load_resolver = resolve;
            });
        }
        //     showPlaybackControls(display_element, trial) {
        //         display_element.innerHTML = `
        //     <p><audio id="playback" src="${this.audio_url}" controls></audio></p>
        //     <button id="record-again" class="jspsych-btn">${trial.record_again_button_label}</button>
        //     <button id="continue" class="jspsych-btn">${trial.accept_button_label}</button>
        //   `;
        //         display_element.querySelector("#record-again").addEventListener("click", () => {
        //             // release object url to save memory
        //             URL.revokeObjectURL(this.audio_url);
        //             this.startRecording();
        //         });
        //         display_element.querySelector("#continue").addEventListener("click", () => {
        //             this.endTrial(display_element, trial);
        //         });
        //         // const audio = display_element.querySelector('#playback');
        //         // audio.src =
        //     }
        endTrial(display_element, trial) {
            // clear recordering event handler
            this.recorder.removeEventListener("dataavailable", this.data_available_handler);
            this.recorder.removeEventListener("start", this.start_event_handler);
            this.recorder.removeEventListener("stop", this.stop_event_handler);
            // kill any remaining setTimeout handlers
            this.jsPsych.pluginAPI.clearAllTimeouts();
            // gather the data to store for the trial
            var trial_data = {
                rt: this.rt,
                stimulus: trial.stimulus,
                response: this.response,
                button_response: this.button,
                chosen_button_id: `jspsych-html-button-response-button-${this.button}`,
                estimated_stimulus_onset: Math.round(this.stimulus_start_time - this.recorder_start_time),
            };
            if (trial.save_audio_url) {
                trial_data.audio_url = this.audio_url;
            }
            else {
                URL.revokeObjectURL(this.audio_url);
            }
            // clear the display
            display_element.innerHTML = "";
            // move on to the next trial
            this.jsPsych.finishTrial(trial_data);
        }
    }
    HtmlAudioButtonResponsePlugin.info = info;
  
    return HtmlAudioButtonResponsePlugin;
  
  })(jsPsychModule);
