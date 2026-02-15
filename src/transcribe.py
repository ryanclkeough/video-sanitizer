from faster_whisper import WhisperModel
import os

INPUT_AUDIO = os.path.join("input", "audio.wav")
OUTPUT_FILE = os.path.join("output", "transcript.txt")

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe(
    INPUT_AUDIO,
    vad_filter=True
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for segment in segments:
        line = f"{segment.start:.2f} {segment.end:.2f} {segment.text}\n"
        f.write(line)

print("Transcription complete.")
