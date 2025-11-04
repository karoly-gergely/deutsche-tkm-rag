#!/usr/bin/env python3
"""Script to rebuild the vector store index."""
import argparse
import shutil
import sys
from pathlib import Path

from config import settings
from monitoring.logging import setup_logging
from scripts.ingest import main as ingest_main

logger = setup_logging()


def main():
    """Main rebuild function."""
    parser = argparse.ArgumentParser(description="Rebuild vector store index")
    parser.add_argument(
        "--data-folder",
        type=str,
        help="Path to data folder (overrides config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild without confirmation",
    )
    args = parser.parse_args()

    chroma_dir = Path(settings.CHROMA_DIR)

    # Check if index exists
    if chroma_dir.exists() and chroma_dir.is_dir():
        if args.force:
            logger.info(f"Removing existing index at {chroma_dir} (forced)")
            shutil.rmtree(chroma_dir)
            print(f"✓ Removed existing index at {chroma_dir}")
        else:
            response = (
                input(f"Delete existing index at {chroma_dir}? [y/N]: ")
                .strip()
                .lower()
            )
            if response == "y":
                logger.info(f"Removing existing index: {chroma_dir}")
                shutil.rmtree(chroma_dir)
                print(f"✓ Removed existing index at {chroma_dir}")
            else:
                print("✗ Rebuild aborted by user.")
                sys.exit(0)
    else:
        print(f"ℹ No existing index found at {chroma_dir}")

    # Prepare arguments for ingestion
    original_argv = sys.argv.copy()
    sys.argv = ["ingest.py"]
    if args.data_folder:
        sys.argv.extend(["--data-folder", args.data_folder])

    try:
        # Run ingestion pipeline
        print("\n" + "=" * 60)
        print("Running ingestion pipeline...")
        print("=" * 60)
        ingest_main()
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    except Exception as e:
        logger.error(f"Error during rebuild: {e}", exc_info=True)
        print(f"✗ Fatal error during rebuild: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
