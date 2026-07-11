"""Backward-compatible command entry point for the synthetic-quality CLI."""

from pyosv.cli.synthetic_quality import main


if __name__ == "__main__":
    raise SystemExit(main())
