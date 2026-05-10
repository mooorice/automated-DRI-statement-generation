"""Allow `python -m autoconcourse` as a shortcut for the pipeline CLI."""

from .cli import pipeline_main

if __name__ == "__main__":
    pipeline_main()
