import argparse
import time
import os
from pathlib import Path

from src.infer import run_inference, save_predictions
from src.config import WINDOWS_LOG_PATH
from src.infer import save_line_predictions


def main():
    parser = argparse.ArgumentParser(description="Run LogBERT inference (placeholder or real)")
    parser.add_argument("--session-size", type=int, default=20, help="events per session")
    parser.add_argument("--stride", type=int, default=10, help="step between sessions")
    parser.add_argument("--no-placeholder", action="store_true", help="use real LogBERT via logbert_infer.py")
    parser.add_argument("--watch", action="store_true", help="watch windows.log and update predictions on change")
    parser.add_argument("--interval", type=float, default=2.0, help="polling interval in seconds when watching")
    args = parser.parse_args()

    def run_once():
        preds, line_preds = run_inference(
            use_placeholder=not args.no_placeholder,
            session_size=args.session_size,
            stride=args.stride,
        )
        out = save_predictions(preds)
        out2 = save_line_predictions(line_preds)
        print(f"Saved predictions to {out} (sessions={len(preds)})")
        print(f"Saved line predictions to {out2}")

    if not args.watch:
        run_once()
        return

    # Watch mode: rerun when file size or mtime changes
    log_path: Path = WINDOWS_LOG_PATH
    last_sig = None
    print(f"Watching {log_path} every {args.interval}s ... Press Ctrl+C to stop.")
    while True:
        try:
            if log_path.exists():
                stat = log_path.stat()
                sig = (stat.st_mtime_ns, stat.st_size)
                if sig != last_sig:
                    last_sig = sig
                    run_once()
            else:
                print("Log file not found; waiting...")
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
