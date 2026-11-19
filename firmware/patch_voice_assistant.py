"""Patch ESPHome's voice_assistant component: ignore a RUN_END that belongs to an earlier pipeline run.

usage: python3 patch_voice_assistant.py <component dir>

Home Assistant sends the events of one run in order (RUN_START ... RUN_END) over one TCP
connection. When the device stops a run and Home Assistant does not get (or not yet
process) the stop, Home Assistant ends that run only when the next start request arrives,
so its RUN_END reaches the device after the new request and before the new RUN_START.
Unpatched ESPHome treats it as the end of the new run and stops the microphone without
telling Home Assistant, which leaves the new run open on Home Assistant's side, and the
next start repeats the pattern. Observed on a ReSpeaker Lite in continuous mode.
"""
import pathlib, sys
d = pathlib.Path(sys.argv[1])
h = d / 'voice_assistant.h'; c = d / 'voice_assistant.cpp'
hs = h.read_text(); cs = c.read_text()
assert 'run_started_' not in hs, 'already patched'
hs = hs.replace('  bool continuous_{false};', '  bool continuous_{false};\n  // Hark patch: RUN_START of the latest start request arrived\n  bool run_started_{true};', 1)
old = '      ESP_LOGD(TAG, "Requesting start");'
assert cs.count(old) == 1
cs = cs.replace(old, old + '\n      this->run_started_ = false;  // Hark patch', 1)
old = '    case api::enums::VOICE_ASSISTANT_RUN_START:\n'
assert cs.count(old) == 1
cs = cs.replace(old, old + '      this->run_started_ = true;  // Hark patch\n', 1)
old = '    case api::enums::VOICE_ASSISTANT_RUN_END: {\n'
assert cs.count(old) == 1
cs = cs.replace(old, old + '''      // Hark patch: a RUN_END before the RUN_START of the latest start request belongs to an
      // earlier run; it must not stop the microphone of the run that is starting now
      if (!this->run_started_ && ((this->state_ == State::START_PIPELINE) ||
                                  (this->state_ == State::STARTING_PIPELINE) ||
                                  (this->state_ == State::STREAMING_MICROPHONE))) {
        ESP_LOGW(TAG, "Ignoring RUN_END of an earlier pipeline run");
        this->defer([this]() { this->end_trigger_.trigger(); });
        break;
      }
''', 1)
old = '''      if (this->continuous_ && this->desired_state_ == State::IDLE) {
        this->idle_trigger_.trigger();'''
assert cs.count(old) == 1
cs = cs.replace(old, old + '\n        this->run_started_ = false;  // Hark patch: a new run begins', 1)
old = '''    this->continuous_ = continuous;
    this->silence_detection_ = silence_detection;
'''
assert cs.count(old) == 1
cs = cs.replace(old, old + '    this->run_started_ = false;  // Hark patch: a new run begins\n', 1)
h.write_text(hs); c.write_text(cs)
print('patched', d)
