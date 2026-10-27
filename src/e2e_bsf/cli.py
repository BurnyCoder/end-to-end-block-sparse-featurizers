"""Global: public executable interface. Local: parse commands and initialize audited runtime.

Sources: https://docs.python.org/3/library/argparse.html;
https://docs.pytorch.org/docs/stable/notes/randomness.html.
"""

import argparse  # Standard-library CLI avoids another configuration framework.
from dataclasses import replace  # CLI overrides remain explicit in saved configs.
import os  # Set cuBLAS determinism before importing CUDA-using modules.
import sys  # Dispatch the independent sweep command family without changing legacy arguments.
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Required for deterministic CUDA GEMMs.


def main():
    """Local: dispatch explicit phases; global: offer one reproducible user workflow."""
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)  # Never search parent/other project env files.
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        from .sweep.cli import main as sweep_main
        return sweep_main(sys.argv[2:])
    import torch
    from .config import Config, CONDITIONS
    from .logging_utils import create_run_logger, log_event
    from . import pipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "report", "run", "verify", "publish"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, default="block_kl")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    config = Config.from_env(args.profile, args.root).validate()
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.lr is not None:
        config = replace(config, lr=args.lr)
    logger, log_path = create_run_logger(args.command, Path(config.root) / "logs")
    torch.set_num_threads(4)  # Leave the workstation responsive and avoid tiny-QR thread overhead.
    torch.use_deterministic_algorithms(True)  # Fail on unsupported nondeterministic operations.
    log_event(logger, "command", {"arguments": vars(args), "config": config.to_dict(),
                                   "torch": torch.__version__, "cuda": torch.version.cuda,
                                   "log": str(log_path)})
    try:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use a compatible Torch/driver build")
        if args.command == "prepare":
            pipeline.prepare(config, logger)
        elif args.command == "train":
            pipeline.train_one(config, args.condition, args.seed, logger)
        elif args.command == "evaluate":
            pipeline.evaluate(config, logger)
        elif args.command == "verify":
            from .verification import verify_real_pipeline
            verify_real_pipeline(config, logger)
        elif args.command == "report":
            from .reporting import build_report
            build_report(config, logger)
        elif args.command == "publish":
            from .publication import publish
            publish(config, logger)
        else:
            pipeline.run(config, logger)
    except Exception:
        logger.exception("Command failed; inspect the complete sanitized log")
        raise SystemExit(1) from None  # The formatter already wrote a redacted traceback.


if __name__ == "__main__":
    main()  # Also support python -m e2e_bsf.cli.
