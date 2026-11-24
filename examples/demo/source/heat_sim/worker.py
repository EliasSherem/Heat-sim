"""Background job process: keeps Streamlit responsive during solver runs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from .storage import atomic_json, snapshot_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    folder = args.request.parent
    request = json.loads(args.request.read_text())
    status_file = folder / "status.json"
    def progress(fraction, message):
        atomic_json(status_file, {"state": "running", "progress": float(fraction), "message": message,
                                   "updated_utc": datetime.now(timezone.utc).isoformat()})
    try:
        snapshot_source(folder)
        progress(0, "Loading numerical model (first run compiles CPU kernels)")
        from .workflows import run_workflow
        from .solver import SimulationCancelled
        if request["kind"] == "verification":
            from .verification import run_verification
            from .storage import RESULTS
            verification = run_verification(include_sensitivity=True, progress=progress,
                cancel=lambda: (folder / "CANCEL").exists())
            atomic_json(RESULTS / "verification.json", verification)
            atomic_json(folder / "report.json", {"kind": "verification", "rows": [], "extra": verification})
            if not verification.get("numerical_checks_passed", False):
                raise ValueError("Numerical verification failed; inspect the saved report")
        else:
            run_workflow(request, folder, progress, lambda: (folder / "CANCEL").exists())
        atomic_json(status_file, {"state": "complete", "progress": 1., "message": "Results saved"})
    except BaseException as exc:
        cancelled = exc.__class__.__name__ == "SimulationCancelled"
        atomic_json(status_file, {"state": "cancelled" if cancelled else "failed", "progress": 0.,
                                   "message": str(exc), "exception": type(exc).__name__})
        if not cancelled:
            (folder / "traceback.txt").write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
