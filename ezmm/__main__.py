"""Command line interface of ezMM. Usage:
    python -m ezmm ui [--path PATH] [--host HOST] [--port PORT]   # Browse the registry in the browser
    python -m ezmm dedup [--path PATH] [--dry-run] [--verbose]    # Remove duplicate files from the registry
    python -m ezmm migrate [--path PATH]                          # Migrate a legacy registry DB
    python -m ezmm check [--path PATH]                            # Check for missing files, refresh sizes
    python -m ezmm embed [--path PATH] [--kind KIND]              # Embed all items (for semantic search)
"""
import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="ezmm", description="ezMM command line tools.")
    parser.add_argument("--path", help="Root directory of the ezMM registry "
                                       "(default: EZMM environment variable or 'temp/').")
    commands = parser.add_subparsers(dest="command", required=True)

    ui = commands.add_parser("ui", help="Start the web UI for browsing the registry.")
    ui.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1).")
    ui.add_argument("--port", type=int, default=7878, help="Port to listen on (default: 7878).")

    dedup = commands.add_parser("dedup", help="Identify identical files and remove the duplicates.")
    dedup.add_argument("--dry-run", action="store_true", help="Only report duplicates, change no items or files "
                                                                "(missing file hashes get computed and saved).")
    dedup.add_argument("--verbose", action="store_true", help="List all removed duplicates (default: the first 20).")

    migrate = commands.add_parser("migrate", help="Migrate a legacy registry DB to the current schema.")

    check = commands.add_parser("check", help="Check for all items whether their file exists "
                                              "and update the registry's 'missing' flags and file sizes.")

    embed = commands.add_parser("embed", help="Compute the embeddings of all items that are not embedded yet "
                                              "(makes them searchable in the web UI).")
    embed.add_argument("--kind", help="Only embed items of this kind (image, video, audio, or file).")

    # Allow `--path` after the sub-command, too
    for sub in (ui, dedup, migrate, check, embed):
        sub.add_argument("--path", dest="sub_path", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    path = getattr(args, "sub_path", None) or args.path

    from ezmm.common import item_registry
    from ezmm.common.items.file import format_size

    if path:
        os.environ["EZMM"] = Path(path).absolute().as_posix()
        item_registry.close()
        item_registry.set_path(path)
    print(f"Using ezMM registry at {item_registry.path.as_posix()}")

    if args.command == "ui":
        from ezmm.ui.main import run_server
        run_server(host=args.host, port=args.port)

    elif args.command == "dedup":
        report = item_registry.deduplicate(dry_run=args.dry_run)  # Shows progress bars
        prefix = "[DRY RUN] Would remove" if args.dry_run else "Removed"
        shown = report["removed"] if args.verbose else report["removed"][:20]
        for kind, dup_id, keeper_id in shown:
            print(f"  <{kind}:{dup_id}> -> <{kind}:{keeper_id}>")
        if len(shown) < len(report["removed"]):
            print(f"  ... and {len(report['removed']) - len(shown)} more (use --verbose to list all)")
        print(f"{prefix} {len(report['removed'])} duplicates in {report['groups']} groups, "
              f"deleting {len(report['deleted_files'])} files ({format_size(report['freed_bytes'])}). "
              f"Hashed {report['hashed']} files that had no hash yet.")

    elif args.command == "migrate":
        item_registry.migrate()
        print(f"Registry at {item_registry.path.as_posix()} is up to date.")

    elif args.command == "check":
        result = item_registry.check_files()
        print(f"Checked {result['checked']} items: {result['missing']} files missing "
              f"({result['changed']} flags updated), {result['sizes_updated']} file sizes updated. "
              f"Total size: {format_size(item_registry.total_size())}.")

    elif args.command == "embed":
        from ezmm.embedding import INSTALL_HINT, MODEL_NAME, embed_registry, is_available
        if not is_available():
            parser.exit(1, INSTALL_HINT + "\n")
        print(f"Embedding {item_registry.count_unembedded(MODEL_NAME)} items with {MODEL_NAME}...")
        result = embed_registry(kind=args.kind)  # Shows a progress bar
        print(f"Embedded {result['embedded']} items ({result['failed']} failed).")


if __name__ == "__main__":
    main()
