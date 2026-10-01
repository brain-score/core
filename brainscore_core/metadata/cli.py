import argparse
from pathlib import Path
from .contract import load, MetadataError
from .policy import protected_changes


def main():
    parser = argparse.ArgumentParser(
        description="Validate model metadata without importing plugin code or connecting to a database."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()
    try:
        document = load(args.path.read_text(), args.domain)
        if args.base:
            base = load(args.base.read_text(), args.domain)
            if set(base["models"]) != set(document["models"]):
                raise MetadataError(
                    "Model additions/removals require the repository registration workflow"
                )
            protected = {
                key: protected_changes(value, document["models"][key])
                for key, value in base["models"].items()
            }
            if any(protected.values()):
                raise MetadataError(
                    f"Protected changes require explicit maintainer review: {protected}"
                )
    except (MetadataError, OSError) as exc:
        parser.error(str(exc))
    print("Metadata is valid.")
