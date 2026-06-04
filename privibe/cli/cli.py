from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rich import print as rprint
import tomli_w

from privibe import __version__
from privibe.cli.textual_ui.app import StartupOptions, run_textual_ui
from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import (
    MissingPromptFileError,
    PATHS_TEMPLATE_FILE,
    VibeConfig,
    load_dotenv_values,
)
from privibe.core.config.harness_files import get_harness_files_manager
from privibe.core.config.harness_files._paths import GLOBAL_SKILLS_DIR
from privibe.core.logger import logger
from privibe.core.paths import HISTORY_FILE
from privibe.core.programmatic import run_programmatic
from privibe.core.session.session_loader import SessionLoader
from privibe.core.types import EntrypointMetadata, OutputFormat
from privibe.core.utils import ConversationLimitException


def _print_available_tools() -> None:
    from privibe.core.tools.manager import ToolManager
    from privibe.core.paths import DEFAULT_TOOL_DIR

    tools = sorted(ToolManager._iter_tool_classes([DEFAULT_TOOL_DIR.path]), key=lambda t: t.get_name())
    max_name = max((len(t.get_name()) for t in tools), default=0)
    rprint("[bold]Available tools:[/]")
    for tool in tools:
        name = tool.get_name()
        desc = tool.description.split(".")[0].strip()
        rprint(f"  [cyan]{name:<{max_name}}[/]  {desc}")
    rprint(f"\n[dim]{len(tools)} tools total. "
           "Use [bold]disabled_tools[/] or [bold]enabled_tools[/] in config.toml to control which are active.[/]")


def get_initial_agent_name(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.agent == BuiltinAgentName.DEFAULT:
        return BuiltinAgentName.AUTO_APPROVE
    return args.agent


def get_prompt_from_stdin() -> str | None:
    if sys.stdin.isatty():
        return None
    try:
        if content := sys.stdin.read().strip():
            sys.stdin = sys.__stdin__ = open("/dev/tty")
            return content
    except KeyboardInterrupt:
        pass
    except OSError:
        return None

    return None


def load_config_or_exit() -> VibeConfig:
    try:
        return VibeConfig.load()
    except MissingPromptFileError as e:
        rprint(f"[yellow]Invalid system prompt id: {e}[/]")
        sys.exit(1)
    except ValueError as e:
        rprint(f"[yellow]{e}[/]")
        sys.exit(1)


def bootstrap_config_files() -> None:
    mgr = get_harness_files_manager()
    config_file = mgr.user_config_file
    if not config_file.exists():
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_data = VibeConfig.create_default()
            # The [paths] section ships from a static template so its inline
            # comments and example aliases survive (tomli_w drops comments).
            # PathConfig in privibe/core/config/_settings.py is the runtime
            # source of truth; a roundtrip test pins the two together.
            #
            # The template MUST be written AFTER tomli_w.dump because it ends
            # with `[paths.aliases]` (an empty sub-table). In TOML, anything
            # following an open table header belongs to that table until the
            # next header — so writing the template first would silently nest
            # every other top-level key under [paths.aliases]. Dump first,
            # template last.
            config_data.pop("paths", None)
            paths_template = PATHS_TEMPLATE_FILE.read_text(encoding="utf-8")
            with config_file.open("wb") as f:
                tomli_w.dump(config_data, f)
                f.write(b"\n")
                f.write(paths_template.encode("utf-8"))
        except Exception as e:
            rprint(f"[yellow]Could not create default config file: {e}[/]")
    else:
        # Existing config: offer commented stubs for any new keys we've added
        # to the schema since the user's last upgrade. Pure documentation —
        # no behavior change unless they explicitly uncomment.
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
            rprint(f"[yellow]Could not create history file: {e}[/]")

    _bootstrap_sample_skills()


def _bootstrap_sample_skills() -> None:
    """Copy bundled sample skills to ~/.privibe/skills/ if they don't already exist."""
    from privibe import VIBE_ROOT

    source_dir = VIBE_ROOT / "sample_skills"
    if not source_dir.is_dir():
        return

    dest_dir = GLOBAL_SKILLS_DIR.path
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        rprint(f"[yellow]Could not create skills directory: {e}[/]")
        return

    for skill_src in source_dir.iterdir():
        if not skill_src.is_dir():
            continue
        skill_dest = dest_dir / skill_src.name
        if skill_dest.exists():
            continue
        try:
            import shutil

            shutil.copytree(skill_src, skill_dest)
        except OSError as e:
            rprint(f"[yellow]Could not install sample skill '{skill_src.name}': {e}[/]")


def load_session(
    args: argparse.Namespace, config: VibeConfig
) -> tuple[dict, Path] | None:
    if not args.continue_session and not args.resume:
        return None

    if not config.session_logging.enabled:
        rprint(
            "[red]Session logging is disabled. "
            "Enable it in config to use --continue or --resume[/]"
        )
        sys.exit(1)

    session_to_load = None
    if args.continue_session:
        session_to_load = SessionLoader.find_latest_session(config.session_logging)
        if not session_to_load:
            rprint(
                f"[red]No previous sessions found in "
                f"{config.session_logging.save_dir}[/]"
            )
            sys.exit(1)
    elif args.resume is True:
        return None
    else:
        session_to_load = SessionLoader.find_session_by_id(
            args.resume, config.session_logging
        )
        if not session_to_load:
            rprint(
                f"[red]Session '{args.resume}' not found in "
                f"{config.session_logging.save_dir}[/]"
            )
            sys.exit(1)

    try:
        _, metadata = SessionLoader.load_session(session_to_load)
        return metadata, session_to_load
    except Exception as e:
        rprint(f"[red]Failed to load session: {e}[/]")
        sys.exit(1)


def _resume_previous_session(
    agent_loop: AgentLoop, metadata: dict, session_path: Path
) -> None:
    session_id = metadata.get("session_id", agent_loop.session_id)
    agent_loop.session_id = session_id
    agent_loop.session_logger.resume_existing_session(session_id, session_path)
    agent_loop.messages.restore(session_path)

    logger.info(
        "Resumed session %s with %d messages", session_id, len(agent_loop.messages)
    )


def run_cli(args: argparse.Namespace) -> None:
    load_dotenv_values()
    bootstrap_config_files()


    if getattr(args, "list_tools", False):
        _print_available_tools()
        sys.exit(0)

    try:
        initial_agent_name = get_initial_agent_name(args)
        config = load_config_or_exit()

        if args.enabled_tools:
            config.enabled_tools = args.enabled_tools

        loaded_session = load_session(args, config)

        stdin_prompt = get_prompt_from_stdin()
        if args.prompt is not None:
            config.disabled_tools = [*config.disabled_tools, "ask_user_question"]
            programmatic_prompt = args.prompt or stdin_prompt
            if not programmatic_prompt:
                print(
                    "Error: No prompt provided for programmatic mode", file=sys.stderr
                )
                sys.exit(1)
            output_format = OutputFormat(
                args.output if hasattr(args, "output") else "text"
            )

            try:
                final_response = run_programmatic(
                    config=config,
                    prompt=programmatic_prompt or "",
                    max_turns=args.max_turns,
                    max_price=args.max_price,
                    output_format=output_format,
                    session_path=loaded_session[1] if loaded_session else None,
                    agent_name=initial_agent_name,
                )
                if final_response:
                    print(final_response)
                sys.exit(0)
            except ConversationLimitException as e:
                print(e, file=sys.stderr)
                sys.exit(1)
            except RuntimeError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            agent_loop = AgentLoop(
                config,
                agent_name=initial_agent_name,
                enable_streaming=True,
                entrypoint_metadata=EntrypointMetadata(
                    agent_entrypoint="cli",
                    agent_version=__version__,
                    client_name="privibe_cli",
                    client_version=__version__,
                ),
            )

            if loaded_session:
                _resume_previous_session(agent_loop, *loaded_session)

            run_textual_ui(
                agent_loop=agent_loop,
                startup=StartupOptions(
                    initial_prompt=args.initial_prompt or stdin_prompt,
                    show_resume_picker=args.resume is True,
                ),
            )

    except (KeyboardInterrupt, EOFError):
        rprint("\n[dim]Bye![/]")
        sys.exit(0)
