"""Global: a separate public workflow; local: one process writer lock and explicit phases.

Sources: argparse, contextlib, msvcrt.locking / fcntl.flock standard-library documentation.
"""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import time
import torch
from ..io import write_json
from ..logging_utils import create_run_logger, log_event
from .spec import SweepSpec


@contextmanager
def writer_lock(root):
    """An OS lock releases on process death, so stale PID files never authorize a second writer."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+b") as handle:
        # File-size inspection does not read a byte that Windows may already have locked.
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another sweep writer is active; do not start a second worker") from error
        try:
            write_json(root / "writer.json", {"active": True, "pid": os.getpid(), "started_unix": time.time()})
            yield
        finally:
            write_json(root / "writer.json", {"active": False, "pid": os.getpid(), "finished_unix": time.time()})
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def main(argv):
    """Keep all credentials in .env and sanitize complete timestamped command/error logs."""
    parser = argparse.ArgumentParser(description="Separately registered DINO block-6 sparsity sweep")
    parser.add_argument("phase", choices=("prepare", "verify", "pilot", "train", "evaluate", "report", "run", "publish"))
    parser.add_argument("--spec", default="configs/downstream-sweep.toml", type=Path)
    parser.add_argument("--cell", help="For train: advance one frozen cell, without changing its recipe")
    args = parser.parse_args(argv)
    spec = SweepSpec.load(args.spec)
    if args.cell and (args.phase != "train" or args.cell not in {cell.key for cell in spec.cells()}):
        parser.error("--cell must name a registered training cell")
    logger, log_path = create_run_logger(f"sweep-{args.phase}", Path(spec.root) / "logs")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    log_event(logger, "sweep_command", {"phase": args.phase, "spec": spec.to_dict(), "log": str(log_path)})
    from .pipeline import run
    try:
        with writer_lock(spec.root):
            if args.phase in ("prepare", "run") and not (Path(spec.root) / "study_status.json").exists():
                from .publication import check_access
                log_event(logger, "sweep_upload_access", check_access())
            run(spec, logger, args.phase, args.cell)
    except Exception:
        logger.exception("Sweep stopped; preserve original deadlines and inspect this complete log")
        raise SystemExit(1) from None
