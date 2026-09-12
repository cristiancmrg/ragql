# SPDX-License-Identifier: MPL-2.0
# src/ragql/config.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field, fields, asdict
import os
import json
import logging
from json import JSONDecodeError

# load_dotenv()

CONFIG_FILE = "rag_config.json"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Settings:
    db_path: Path = Path(__file__).parent / ".ragql.db"
    chunk_size: int = 800
    chunk_overlap: int = 80

    # keys / endpoints
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    ollama_url: str = os.getenv("OLLAMA_URL", "")
    use_ollama: bool = False

    # user prefs
    verbose = False
    allowed_folders: list[str] = field(default_factory=list)
    line_spacing: int = 1
    response_color: str = "default"

    # embed model selection: "<provider>::<model-name>"
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "RAGQL_EMBED_MODEL", "openai::text-embedding-ada-002"
        )
    )

    @property
    def embed_provider(self) -> str:
        """Return the embedding provider (before the '::')."""
        return self.embed_model.split("::", 1)[0]

    @property
    def embed_model_name(self) -> str:
        """Return the embedding model name (after the '::')."""
        parts = self.embed_model.split("::", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    @classmethod
    def load(cls) -> Settings:
        """Read JSON config if present; otherwise fall back to env once."""
        logger.debug("Starting Settings.load()")
        cfg = cls()
        p = Path(CONFIG_FILE)
        if p.exists():
            logger.debug("Found config file at %s", p)
            text = p.read_text()
            if not text.strip():
                logger.info("Config file is empty—using defaults.")
            else:
                try:
                    data = json.loads(text) if text.strip() else {}
                    logger.debug("Parsed JSON config: %r", data)
                except JSONDecodeError:
                    logger.warning(
                        "Invalid JSON in %r—falling back to defaults.", CONFIG_FILE
                    )
                    return cfg
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                        logger.info(f"loaded config {k!r} → {v!r}")
                    else:
                        logger.debug("Skipping unknown config key %r", k)
        else:
            logger.info(
                "No config file found at %r—loading from environment variables.",
                CONFIG_FILE,
            )

            # first-time run: grab any exisitng env vars

            cfg.openai_key = os.getenv("OPENAI_API_KEY", "")
            cfg.ollama_url = os.getenv("OLLAMA_URL", "")
            cfg.use_ollama = bool(cfg.ollama_url)

            logger.debug(
                "Env-loaded: openai_key=%r, ollama_url=%r, use_ollama=%r",
                cfg.openai_key,
                cfg.ollama_url,
                cfg.use_ollama,
            )

        return cfg

    def save(self) -> None:
        """Write current settings back to JSON (including embed_model)."""
        logger.debug("Saving settings to %s", CONFIG_FILE)

        data = asdict(self)
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Config successfully written (%d keys).", len(data))


def config_menu() -> None:
    cfg = Settings.load()
    while True:
        print("\nConfiguration Menu:")
        print("1. Customize Line Spacing")
        print("2. Customize Response Color")
        print("3. Set OpenAI API Key")
        print("4. Toggle verbose mode by default")
        print("5. Save and Exit")
        print("6. Exit without Saving")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            val = input("Enter line spacing (e.g., 1, 2, 3): ").strip()
            try:
                cfg.line_spacing = int(val)
                print(f"Line spacing set to {cfg.line_spacing}.")
            except ValueError:
                print("Invalid number. Please enter an integer.")
        elif choice == "2":
            color = input(
                "Enter response color (e.g., 'red','green','blue','default'): "
            ).strip()
            cfg.response_color = color
            print(f"Response color set to '{cfg.response_color}'.")
        elif choice == "3":
            key = input("Enter your OpenAI API key: ").strip()
            cfg.openai_key = key
            print("OpenAI API key update in config.")
        elif choice == "4":
            option = input(
                "Enter True to enable verbose by default and False to disable it"
            )
            cfg.verbose = bool(option)
            if option:
                print("Verbose mode set to enabled")
            else:
                print("Verbose mode set to disable")
        elif choice == "5":
            cfg.save()
            print("Configuration saved.")
            break
        elif choice == "6":
            print("Exiting without saving.")
            break
        else:
            print("Invalid choice. Please try again.")


def add_config_file() -> None:
    """Create a fresh config file with all defaults."""
    default = Settings()
    default.save()
    logger.info("Default config written to %s", CONFIG_FILE)


def add_folder(folder: str) -> None:
    """Append a folder to allowed_folders (unless it’s already there)."""
    cfg = Settings.load()
    if folder in cfg.allowed_folders:
        logger.warning("Folder '%s' already present in allowed_folders", folder)
        return

    cfg.allowed_folders.append(folder)
    cfg.save()  # uses Settings.save(), which itself logs

    logger.info(
        "Added '%s' to allowed_folders (now %d entries)",
        folder,
        len(cfg.allowed_folders),
    )


def set_openai_key(new_key: str) -> None:
    """Update the OpenAI key in rag_config.json."""

    cfg = Settings.load()
    old = cfg.openai_key
    cfg.openai_key = new_key
    cfg.save()

    logger.info("OPENAI_API_KEY updated from %r to %r in %s", old, new_key, CONFIG_FILE)


def migrate_config() -> None:
    """
    Read the old config file, transform only the
    pieces that need updating, preserve everything else,
    and write back the merged result.
    """
    p = Path(CONFIG_FILE)
    if not p.exists():
        logger.info("No config file at %s—nothing to migrate", CONFIG_FILE)
        return

    logger.debug("Starting migration of %s", CONFIG_FILE)

    # Build a Settings instance from it (ignores unknown keys)
    old_cfg = Settings.load()  # your existing loader

    # Create an up-to-date “base” dataclass
    new_cfg = Settings()  # this has all the new defaults

    # Copy over any fields that existed in old_cfg
    # (this preserves user-set values, including fields you didn’t touch in the new schema)
    for f in fields(Settings):
        name = f.name
        if hasattr(old_cfg, name):
            value = getattr(old_cfg, name)
            setattr(new_cfg, name, value)
            logger.debug("Preserved %s = %r", name, value)

    # Now apply the schema changes.
    # For example, say you renamed `foo` → `bar`, or split `baz` into two:
    # new_cfg.bar = new_cfg.foo; delattr(new_cfg, "foo")
    # new_cfg.new_field = compute_something(old_cfg.some_other_field)

    # (Only mutate the pieces that actually changed in the new version!)

    # Dump the merged result back to JSON
    new_cfg.save()
    logger.info("Config migrated successfully—saved to %s", CONFIG_FILE)
