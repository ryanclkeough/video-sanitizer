import csv
import os
from dataclasses import dataclass

IN_CSV = os.path.join("output", "detections.csv")
OUT_CSV = os.path.join("output", "detections_merged.csv")

MERGE_GAP_SEC = 1.5  # merge if within this gap

@dataclass
class Row:
    start: float
    end: float
    reason: str
    confidence: float
    snippet: str

def sec_to_hms(s: float) -> str:
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

rows = []
with open(IN_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for d in r:
        rows.append(Row(
            start=float(d["start_sec"]),
            end=float(d["end_sec"]),
            reason=d["reason"],
            confidence=float(d["confidence"]),
            snippet=d["snippet"],
        ))

rows.sort(key=lambda x: (x.start, x.end))

merged = []
for row in rows:
    if not merged:
        merged.append(row)
        continue

    last = merged[-1]
    same_reason = last.reason == row.reason
    close_enough = row.start <= last.end + MERGE_GAP_SEC

    if same_reason and close_enough:
        last.end = max(last.end, row.end)
        last.confidence = max(last.confidence, row.confidence)
        if row.snippet and row.snippet not in last.snippet:
            last.snippet = (last.snippet + " | " + row.snippet)[:240]
    else:
        merged.append(row)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["start_sec", "end_sec", "start_hms", "end_hms", "reason", "confidence", "snippet"])
    for m in merged:
        w.writerow([m.start, m.end, sec_to_hms(m.start), sec_to_hms(m.end), m.reason, m.confidence, m.snippet])

print(f"Merged {len(rows)} -> {len(merged)} rows into {OUT_CSV}")
