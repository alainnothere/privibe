from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys

import tomli_w

from privibe import __version__
from privibe.core.config import PATHS_TEMPLATE_FILE, MissingAPIKeyError, VibeConfig
from privibe.core.config.harness_files import (
    get_harness_files_manager,
    init_harness_files_manager,
)
from privibe.core.logger import logger
from privibe.core.paths import HISTORY_FILE

# Configure line buffering for subprocess communication
sys.stdout.reconfigure(line_buffering=True)  # pyright: ignore[reportAttributeAccessIssue]
sys.stderr.reconfigure(line_buffering=True)  # pyright: ignore[reportAttributeAccessIssue]
sys.stdin.reconfigure(line_buffering=True)  # pyright: ignore[reportAttributeAccessIssue]


@dataclass
class Arguments:
    pass

def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description="Run Privibe in ACP mode")
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args()
    return Arguments()


def bootstrap_config_files() -> None:
    mgr = get_harness_files_manager()
    config_file = mgr.user_config_file
    if not config_file.exists():
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_data = VibeConfig.create_default()
            # See privibe/cli/cli.py:bootstrap_config_files for why the template
            # must be written AFTER the dump (TOML scope leak via
            # [paths.aliases]). Keep these two bootstrap paths in sync.
            config_data.pop("paths", None)
            paths_template = PATHS_TEMPLATE_FILE.read_text(encoding="utf-8")
            with config_file.open("wb") as f:
                tomli_w.dump(config_data, f)
                f.write(b"\n")
                f.write(paths_template.encode("utf-8"))
        except Exception as e:
            logger.error(f"Could not create default config file: {e}")
            raise
    else:
        # Same migration check as the CLI bootstrap; see privibe/cli/cli.py.
        try:
            from privibe.core.config.migration import run_migration_if_needed
            run_migration_if_needed(config_file)
        except Exception as e:
            logger.warning("Config migration check failed: %s", e)

    history_file = HISTORY_FILE.path
    if not history_file.exists():
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            history_file.write_text("Hello Privibe!\n", "utf-8")
        except Exception as e:
            logger.error(f"Could not create history file: {e}")
            raise


def handle_debug_mode() -> None:
    if os.environ.get("DEBUG_MODE") != "true":
        return

    try:
        import debugpy
    except ImportError:
        return

    debugpy.listen(("localhost", 5678))
    # uncomment this to wait for the debugger to attach
    # debugpy.wait_for_client()


def main() -> None:
    handle_debug_mode()
    init_harness_files_manager("user", "project")

    from privibe.acp.acp_agent_loop import run_acp_server
    from privibe.core.config import VibeConfig, load_dotenv_values

    load_dotenv_values()
    bootstrap_config_files()
    args = parse_arguments()

    run_acp_server()


if __name__ == "__main__":
    main()
