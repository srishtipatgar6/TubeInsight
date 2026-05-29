"""Optional trainer for replacing the shipped Tube Insights weights.

Usage:
  python train_model.py data/videos.csv

CSV columns expected:
  title,description,likes,comments,subscribers,is_viral

The app is deliberately runnable without this script. Use it once you have real
channel or public benchmark data, then replace the weights in tubeinsight/engine.py
or extend the loader to read the generated JSON.
"""

import csv
import json
import math
import sys
from pathlib import Path


FEATURES = [
    "engagement_rate",
    "comment_intensity",
    "title_length",
    "description_length",
    "subscriber_scale",
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Pass a CSV path: python train_model.py data/videos.csv")
    rows = load_rows(Path(sys.argv[1]))
    weights = train_logistic(rows)
    output = Path("trained_weights.json")
    output.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(f"Wrote {output.resolve()}")


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def vector(row):
    likes = float(row.get("likes") or 0)
    comments = float(row.get("comments") or 0)
    subscribers = max(float(row.get("subscribers") or 0), 100)
    return [
        min((likes + comments * 2.2) / subscribers, 1.8),
        min(comments / max(likes, 1), 0.65),
        min(len(row.get("title", "")) / 80, 1.5),
        min(len(row.get("description", "")) / 1200, 1.5),
        min(math.log10(max(subscribers, 1)) / 7, 1),
    ]


def train_logistic(rows, epochs=1800, lr=0.08):
    weights = [0.0 for _ in FEATURES]
    bias = 0.0
    for _ in range(epochs):
        for row in rows:
            x = vector(row)
            y = 1.0 if str(row.get("is_viral", "")).lower() in {"1", "true", "yes"} else 0.0
            pred = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
            error = pred - y
            weights = [w - lr * error * v for w, v in zip(weights, x)]
            bias -= lr * error
    return {"features": FEATURES, "weights": weights, "bias": bias}


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


if __name__ == "__main__":
    main()
