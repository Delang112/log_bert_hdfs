from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import MODEL_DIR, PREDICTIONS_PATH, ensure_paths, LINE_PREDICTIONS_PATH
from .preprocess import read_windows_log, to_log_keys, make_sessions


@dataclass
class Prediction:
    session_index: int
    start_line: int
    end_line: int
    anomaly_score: float
    is_anomaly: bool


@dataclass
class LinePrediction:
    line_number: int
    anomaly_score: float


def _try_load_threshold(default_threshold: float = 0.5) -> float:
    thr_path = MODEL_DIR / "threshold.json"
    if thr_path.exists():
        try:
            obj = json.loads(thr_path.read_text(encoding="utf-8"))
            return float(obj.get("anomaly_threshold", default_threshold))
        except Exception:
            return default_threshold
    return default_threshold


def _fatal_missing_model() -> None:
    raise RuntimeError(
        "LogBERT model code not found. Please provide the inference module that matches 'best_bert.pth' and 'vocab.pkl', or approve dependency installation so I can fetch the proper implementation."
    )


def score_sessions_with_placeholder(sessions: List[List[str]]) -> List[float]:
    # Placeholder scoring: longer/more repetitive sessions get higher scores
    scores: List[float] = []
    for sess in sessions:
        unique_ratio = len(set(sess)) / max(1, len(sess))
        # Heuristic: lower uniqueness => higher anomaly score
        score = max(0.0, min(1.0, 1.0 - unique_ratio))
        scores.append(score)
    return scores


def run_inference(use_placeholder: bool = True, session_size: int = 20, stride: int | None = None) -> tuple[list[Prediction], list[LinePrediction]]:
    ensure_paths()

    lines = read_windows_log()
    keys = to_log_keys(lines)
    sessions = make_sessions(keys, session_size=session_size, stride=stride)

    threshold = _try_load_threshold(0.5)

    if use_placeholder:
        scores = score_sessions_with_placeholder(sessions)
    else:
        # Hook: try to import user-provided logbert inference
        try:
            from logbert_infer import load_model_and_vocab, score_sessions  # type: ignore
        except Exception as e:
            _fatal_missing_model()
            raise e

        model, vocab = load_model_and_vocab(MODEL_DIR)
        scores = score_sessions(model, vocab, sessions)

    preds: List[Prediction] = []
    # Map sessions back to line ranges using stride logic
    # For simplicity, assume stride == session_size when None
    if stride is None or stride <= 0:
        stride = session_size

    line_cursor = 0
    for idx, sess in enumerate(sessions):
        sess_len = len(sess)
        score = float(scores[idx])
        is_anom = score >= threshold

        start_line = (idx * stride) + 1
        end_line = min(start_line + sess_len - 1, len(keys))

        preds.append(
            Prediction(
                session_index=idx,
                start_line=start_line,
                end_line=end_line,
                anomaly_score=score,
                is_anomaly=is_anom,
            )
        )

    # Build line-level scores as max over overlapping session scores
    line_scores = [0.0] * len(keys)
    for p in preds:
        for ln in range(p.start_line, p.end_line + 1):
            idx0 = ln - 1
            if 0 <= idx0 < len(line_scores):
                line_scores[idx0] = max(line_scores[idx0], p.anomaly_score)

    line_preds: list[LinePrediction] = [
        LinePrediction(line_number=i + 1, anomaly_score=float(s)) for i, s in enumerate(line_scores)
    ]

    return preds, line_preds


def save_predictions(preds: List[Prediction]) -> Path:
    data = [p.__dict__ for p in preds]
    PREDICTIONS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return PREDICTIONS_PATH


def save_line_predictions(line_preds: list[LinePrediction]) -> Path:
    data = [lp.__dict__ for lp in line_preds]
    LINE_PREDICTIONS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return LINE_PREDICTIONS_PATH


if __name__ == "__main__":
    preds, line_preds = run_inference(use_placeholder=True)
    out = save_predictions(preds)
    out2 = save_line_predictions(line_preds)
    print(f"Saved predictions to {out}")
    print(f"Saved line predictions to {out2}")
