import re
import csv
import os
import spacy
from detoxify import Detoxify

TRANSCRIPT = os.path.join("output", "transcript.txt")
OUT_CSV = os.path.join("output", "detections.csv")

PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ADDRESS = re.compile(
    r"\b\d{1,5}\s+\w+(?:\s+\w+){0,4}\s+(?:St|Street|Rd|Road|Ave|Avenue|Blvd|Boulevard|Ln|Lane|Dr|Drive)\b",
    re.IGNORECASE,
)

nlp = spacy.load("en_core_web_sm")
tox = Detoxify("original")

def parse_transcript(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if len(parts) < 3:
                continue
            yield float(parts[0]), float(parts[1]), parts[2]

def sensitive_hits(text):
    hits = []
    if PHONE.search(text): hits.append("phone")
    if EMAIL.search(text): hits.append("email")
    if ADDRESS.search(text): hits.append("address")

    doc = nlp(text)
    labels = {e.label_ for e in doc.ents}

    if "PERSON" in labels: hits.append("person")
    if "ORG" in labels: hits.append("org")
    if "GPE" in labels: hits.append("location")

    return hits

def tox_score(text):
    s = tox.predict(text)
    return float(max(s.get("toxicity", 0), s.get("identity_attack", 0), s.get("threat", 0)))

detections = []

for start, end, text in parse_transcript(TRANSCRIPT):
    t = tox_score(text)
    hits = sensitive_hits(text)

    if t >= 0.70:
        detections.append((start, end, "toxicity/identity/threat", round(t, 3), text[:140]))

    if hits:
        detections.append((start, end, "sensitive:" + ",".join(hits), 0.95, text[:140]))

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["start_sec", "end_sec", "reason", "confidence", "snippet"])
    writer.writerows(detections)

print(f"Wrote {len(detections)} detections to {OUT_CSV}")
