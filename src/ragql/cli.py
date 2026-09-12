# SPDX-License-Identifier: MPL-2.0
# src/ragql/cli.py
import argparse
import pathlib
import logging
from .config import (
    Settings,
    config_menu,
    add_config_file,
    add_folder,
    migrate_config,
    set_openai_key,
)
from .core import RagQL

logger = logging.getLogger(__name__)


def main() -> None:
    # Pre-parser: catch only -v/--verbose
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug output"
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Configure logging based on verbose flag
    # Only show debug & info logs when verbose; otherwise show warnings and above
    log_level = logging.DEBUG if pre_args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if pre_args.verbose:
        logger.debug("Verbose mode ON")

    # Load settings (reads .env + config.json)
    cfg = Settings.load()
    logging.debug(f"Loaded config: {cfg!r}")

    ap = argparse.ArgumentParser(
        prog="ragql",
        description="Modular RAG chat over logs & DBs",
    )
    ap.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate your config.json to the new schema, preserving unchanged fields",
    )
    ap.add_argument(
        "--query",
        "-q",
        metavar="QUESTION",
        help="Run one RAG-powered query and exit",
        type=str,
    )
    ap.add_argument(
        "--sources",
        nargs="*",
        help="One or more folders/text files/Data.db files to index",
    )
    ap.add_argument(
        "--remote",
        action="store_true",
        help="Force OpenAI even if OLLAMA_URL is set",
    )
    ap.add_argument(
        "--configs",
        action="store_true",
        help="Enter configuration mode",
    )
    ap.add_argument(
        "command",
        nargs="?",
        help="Command to execute (e.g., 'add', 'add-folder', 'set')",
    )
    ap.add_argument(
        "key_value",
        nargs="*",
        help="Key/value for setting commands (e.g., 'openai key sk-…')",
    )

    # Parse only the leftover args, so -v is not considered here
    args = ap.parse_args(remaining_argv)

    if args.migrate:
        logging.info("Migrating rag_config.json to the new format...")
        migrate_config()
        print("✅ Migration complete!")
        return

    logging.debug(f"Arguments: {args!r}")

    if args.remote:
        cfg.use_ollama = False
        logging.debug("Forcing OpenAI (use_ollama=False)")

    # Subcommands
    if args.command == "add":
        logging.info("Adding new config file")
        add_config_file()
        return

    # Add folder as source to the config
    if args.command == "add-folder" and args.key_value:
        logging.info(f"Adding folder: {args.key_value[0]}")
        add_folder(args.key_value[0])
        return

    # Set openai key
    if args.command == "set" and len(args.key_value) >= 3:
        # Expect: set openai key <API_KEY
        if args.key_value[0] == "openai" and args.key_value[1] == "key":
            logging.info("Setting OpenAI API key")
            set_openai_key(args.key_value[2])
        else:
            ap.error("Usage: ragql set openai key <YOUR_KEY>")
        return

    # Enter in the configs menu
    if args.configs:
        logging.info("Entering configuration menu")
        config_menu()
        return

    sources = args.sources or cfg.allowed_folders

    # Default: need at least one source
    if not sources:
        ap.print_usage()
        print("Please provide at least one --sources path.")
        return

    # Build index from every source you passed (or every allowed_folder)
    paths = [pathlib.Path(s).expanduser().resolve() for s in sources]

    # build the first one
    logging.debug(f"Indexing source: {paths[0]}")
    rq = RagQL(paths[0], cfg)
    rq.build()

    # build the rest re-using the same instance
    for path in paths[1:]:
        logging.debug(f"Indexing source: {path}")
        rq.root = path
        rq.build()

    # At this point `rq` holds the last-built RagQL instance (with your full DB/FAISS index loaded)

    # one-off query via --query
    if args.query:
        logging.info(f"Querying: {args.query}")
        # if you want multi-word, either require quotes:
        #   ragql --query "what is status?"
        # or accept nargs='+' and join them:
        #   ap.add_argument("--query","-q", nargs="+", ...)
        answer = rq.query(args.query)
        print(answer)
        return

    # If they passed an inline question, answer and exit
    # (you could detect more than one and loop, but this matches your old style)
    if args.command is None and len(args.sources) > 1:
        # no command, two positional args: sources + question
        question = args.sources[1]
        logging.info(f"Inline query: {question}")
        print(rq.query(question))
        return

    # Otherwise, drop into REPL:
    print("Entering interactive chat (Ctrl-C to exit)")
    logging.info("Entering interactive chat (Ctrl-C to exit)")

    try:
        while True:
            q = input(">> ").strip()
            if not q:
                continue
            print(rq.query(q))
    except (KeyboardInterrupt, EOFError):
        print()  # newline
        logging.info("Exiting")
        return


if __name__ == "__main__":
    main()
