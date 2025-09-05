"""
Pluggable LogBERT inference adapter.

Implement the two functions below so that src/infer.py can call into them.

Expected behavior:
- load_model_and_vocab(model_dir):
    - Load model weights from f"{model_dir}/best_bert.pth"
    - Load vocab from f"{model_dir}/vocab.pkl"
    - Return (model, vocab)

- score_sessions(model, vocab, sessions):
    - Accept a list of sessions (each session is a list of log keys/ids)
    - Return a list of anomaly scores in [0,1] per session

Tip: This checkpoint appears trained for HDFS (per run_meta.json). If you intend
to score Windows logs, ensure the vocab and model are compatible, or provide a
Windows-trained checkpoint.
"""

from typing import Any, List, Tuple
from pathlib import Path


def load_model_and_vocab(model_dir: Path) -> Tuple[Any, Any]:
    raise NotImplementedError(
        "Please implement LogBERT loading here (depends on your training code)."
    )


def score_sessions(model: Any, vocab: Any, sessions: List[List[str]]) -> List[float]:
    raise NotImplementedError(
        "Please implement LogBERT scoring here (depends on your training code)."
    )

