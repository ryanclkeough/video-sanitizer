from faster_whisper import WhisperModel
import os

INPUT_AUDIO = os.path.join("input", "audio.wav")
OUTPUT_FILE = os.path.join("output", "transcript.txt")

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe(
    INPUT_AUDIO,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=300),
    beam_size=12,
    best_of=12,
    condition_on_previous_text=False, 
    temperature=[0.0, 0.2, 0.4],
    no_speech_threshold=0.1,
    log_prob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for segment in segments:
        line = f"{segment.start:.2f} {segment.end:.2f} {segment.text}\n"
        f.write(line)

print("Transcription complete.")
