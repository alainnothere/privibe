from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from rich import print as rprint
import tomli_w

from privibe import __version__
from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import (
    PATHS_TEMPLATE_FILE,
    MissingPromptFileError,
    VibeConfig,
    load_dotenv_values,
)
from privibe.core.config.harness_files import get_harness_files_manager
from privibe.core.config.harness_files._paths import GLOBAL_SKILLS_DIR
from privibe.core.logger import logger
from privibe.core.paths import HISTORY_FILE
from privibe.core.programmatic import (
    CONTINUE_PROMPT,
    ProgrammaticRunTerminated,
    run_programmatic,
)
from privibe.core.session.session_loader import SessionLoader
from privibe.core.types import EntrypointMetadata, OutputFormat
from privibe.core.utils import ConversationLimitException


def _print_available_tools() -> None:
    from privibe.core.paths import DEFAULT_TOOL_DIR
    from privibe.core.tools.manager import ToolManager

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
    # -p answers no approval prompt, so it runs on auto-approve unless told
    # otherwise. With --nanny there is an approver: keep the asking agent, or
    # the nanny would have nothing to review.
    if (
        args.prompt is not None
        and args.agent == BuiltinAgentName.DEFAULT
        and not getattr(args, "nanny", None)
    ):
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


def load_config_or_exit(**overrides: Any) -> VibeConfig:
    try:
        return VibeConfig.load(**overrides)
    except MissingPromptFileError as e:
        rprint(f"[yellow]Invalid system prompt id: {e}[/]")
        sys.exit(1)
    except ValueError as e:
        rprint(f"[yellow]{e}[/]")
        sys.exit(1)


def validate_model_override(config: VibeConfig, alias: str) -> str | None:
    """Check that a --model override was honored by config loading.

    Returns an error message when the alias is unknown or was rejected
    at load time because its provider API key is missing (VibeConfig
    silently falls back to another model in that case). Returns None
    when the override is in effect.
    """
    matched = next((m for m in config.models if m.alias == alias), None)
    if matched is None:
        available = ", ".join(m.alias for m in config.models)
        return (
            f"Error: Unknown model alias '{alias}'. "
            f"Available models: {available}"
        )
    if config.active_model != alias:
        status = config._is_environment_variable_set_for_model(matched)
        return (
            f"Error: Model '{alias}' is not usable: environment variable "
            f"{status.env_var_name} for provider '{status.provider_name}' "
            f"is not set."
        )
    return None


def load_config_with_cli_overrides(
    model_alias: str | None,
    nanny_alias: str | None,
    nanny_brief: str | None = None,
    *,
    programmatic: bool = False,
) -> VibeConfig:
    """Load the config with --model / --nanny / --nanny-brief applied, or
    exit on an alias that cannot run. No override is written back to
    config.toml.
    """
    if nanny_brief and programmatic and not nanny_alias:
        print(
            "Error: --nanny-brief needs --nanny in programmatic mode (-p): "
            "without one nothing is reviewed.",
            file=sys.stderr,
        )
        sys.exit(1)
    config_overrides: dict[str, Any] = {}
    if model_alias:
        config_overrides["active_model"] = model_alias
    if nanny_alias:
        config_overrides["nanny_model"] = nanny_alias
    if nanny_brief and nanny_brief.strip():
        config_overrides["nanny_brief"] = nanny_brief.strip()
    config = load_config_or_exit(**config_overrides)
    if model_alias and (error := validate_model_override(config, model_alias)):
        print(error, file=sys.stderr)
        sys.exit(1)
    if nanny_alias and (reason := config.unusable_model_reason(nanny_alias)):
        print(f"Error: --nanny: {reason}", file=sys.stderr)
        sys.exit(1)
    return config


def bootstrap_config_files() -> None:
    from privibe.core.paths import legacy_home_env_notice

    if notice := legacy_home_env_notice():
        rprint(f"[yellow]{notice}[/]")

    mgr = get_harness_files_manager()
    config_file = mgr.user_config_file
    if not config_file.exists():
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_data = VibeConfig.create_default()
            # The [paths] section normally ships from a static template so its
            # inline comments and example aliases survive (tomli_w drops
            # comments). PathConfig in privibe/core/config/_settings.py is the
            # runtime source of truth; a roundtrip test pins the two together.
            #
            # If the template can't be read — e.g. a packaging build that
            # failed to bundle default_config.toml (see privibe.spec /
            # privibe-acp.spec) — we must still write a valid config.toml, so
            # fall back to dumping the runtime `paths` defaults inline. The
            # only loss is the explanatory comments; config creation must never
            # hinge on documentation.
            try:
                paths_template: str | None = PATHS_TEMPLATE_FILE.read_text(
                    encoding="utf-8"
                )
            except OSError as e:
                logger.warning(
                    "Paths template %s could not be read (%s); writing "
                    "config.toml with inline path defaults instead.",
                    PATHS_TEMPLATE_FILE,
                    e,
                )
                paths_template = None
            # When the template is present, pop `paths` and append the template
            # LAST: it ends with `[paths.aliases]` (an open sub-table), and in
            # TOML anything after an open table header nests under it — so any
            # top-level key written after it would be captured. When there's no
            # template we keep `paths` in the dump (tomli_w orders tables
            # safely).
            if paths_template is not None:
                config_data.pop("paths", None)
            with config_file.open("wb") as f:
                tomli_w.dump(config_data, f)
                if paths_template is not None:
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

    # Seed / refresh the locally editable prompt copies in ~/.privibe/prompts.
    try:
        from privibe.core.prompts.sync import sync_default_prompts
        sync_default_prompts()
    except Exception as e:
        rprint(f"[yellow]Could not sync default prompts: {e}[/]")


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
    # --continue: THE CONTRACT. Read this before "fixing" it.
    #
    # --continue resumes THE MOST RECENTLY WRITTEN SESSION ON THIS MACHINE.
    # Not "the session I just closed", not "the session of this terminal",
    # not "the session of this directory". The most recent one, where recent
    # means the newest mtime on a messages.jsonl in the session folder
    # (SessionLoader.find_latest_session -> latest_session).
    #
    # What follows from that, all of it intended:
    #   - The session folder is one global directory (logs/session under the
    #     privibe home), shared by every project. There is no filter on the
    #     working directory. A session from another repo can be "the last".
    #   - -p runs save sessions in the same folder. An agent delegating
    #     through `privibe -p` while you were away is a candidate too.
    #   - With two sessions open, the one that wrote last wins. Close an
    #     idle session, run --continue, and you get the busy one. Close ten
    #     and --continue hands you whichever of the ten wrote last.
    #   - Sub-agents and the nanny do NOT count: they save under the parent
    #     session's own folder (agents/, nanny/) with another prefix and
    #     never touch the parent's messages.jsonl.
    #
    # Why it is not "this session": nothing can make that true. A process
    # that exits knows its own id, but --continue is typed into a NEW
    # process that knows nothing about who exited last, and any marker left
    # on disk ("last closed = X") is overwritten by the next session that
    # exits. One session ever, opened and closed in strict sequence, is the
    # only case where "last written" and "the one I meant" must agree.
    #
    # The exit message prints two lines. `privibe --resume <id>` is the one
    # that is always true: the id comes from the exiting process's own
    # memory (resumable_session_id), no shared state, verified with two
    # concurrent sessions on 2026-11-06. `privibe --continue` is the
    # convenience for the common single-session case. Want a specific
    # session back: use the id. This was looked at and deliberately left
    # alone; there is no bug here to find.
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


def programmatic_resume_flags(args: argparse.Namespace) -> list[str]:
    """The flags a -p run must be given again when it is resumed, because
    the session does not bring them back: without --nanny a resumed
    nanny-reviewed run would fall back to auto-approve.
    """
    flags: list[str] = []
    if args.agent != BuiltinAgentName.DEFAULT:
        flags += ["--agent", args.agent]
    if nanny := getattr(args, "nanny", None):
        flags += ["--nanny", nanny]
    if brief := (getattr(args, "nanny_brief", None) or "").strip():
        flags += ["--nanny-brief", brief]
    return flags


def _nanny_saved_with_the_session(
    args: argparse.Namespace, config: VibeConfig, metadata: dict, agent_name: str
) -> tuple[str | None, str]:
    """A resumed session that was reviewed by a nanny stays reviewed: the
    alias in its meta.json is used when --nanny was not given, and the run
    keeps the asking agent, as with the flag. A saved nanny that cannot run
    stops the resume; going on without it would mean auto-approve.
    Returns (nanny alias, agent name).
    """
    saved = metadata.get("nanny_model")
    if not saved:
        return None, agent_name
    if reason := config.unusable_model_reason(saved):
        print(
            f"Error: this session was reviewed by the nanny '{saved}', but "
            f"{reason}. Pass --nanny ALIAS to resume it with another one.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.agent == BuiltinAgentName.DEFAULT:
        agent_name = BuiltinAgentName.DEFAULT
    return saved, agent_name


def _run_programmatic_mode(
    args: argparse.Namespace,
    config: VibeConfig,
    loaded_session: tuple[dict, Path] | None,
    stdin_prompt: str | None,
    *,
    agent_name: str,
    model_alias: str | None,
    nanny_alias: str | None,
) -> None:
    """-p: run one prompt and exit. Never returns."""
    config.disabled_tools = [*config.disabled_tools, "ask_user_question"]
    if not nanny_alias and loaded_session:
        nanny_alias, agent_name = _nanny_saved_with_the_session(
            args, config, loaded_session[0], agent_name
        )
    # `-p --resume ID "text"` parses "text" as the positional prompt; a
    # resumed session with no prompt at all is told to continue.
    prompt = (
        args.prompt
        or getattr(args, "initial_prompt", None)
        or stdin_prompt
        or (CONTINUE_PROMPT if loaded_session else None)
    )
    if not prompt:
        print("Error: No prompt provided for programmatic mode", file=sys.stderr)
        sys.exit(1)
    output_format = OutputFormat(args.output if hasattr(args, "output") else "text")

    try:
        final_response = run_programmatic(
            config=config,
            prompt=prompt,
            max_turns=args.max_turns,
            max_price=args.max_price,
            output_format=output_format,
            session_path=loaded_session[1] if loaded_session else None,
            agent_name=agent_name,
            nanny_alias=nanny_alias,
            resume_flags=programmatic_resume_flags(args),
            pinned=frozenset(
                part
                for part, given in (
                    ("model", bool(model_alias)),
                    ("agent", args.agent != BuiltinAgentName.DEFAULT),
                )
                if given
            ),
        )
    except ConversationLimitException as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ProgrammaticRunTerminated as e:
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        # 130, not the interactive "Bye!" and 0: a script must be able to
        # tell an interrupted run from a finished one.
        sys.exit(130)
    if final_response:
        print(final_response)
    sys.exit(0)


def run_cli(args: argparse.Namespace) -> None:
    load_dotenv_values()
    bootstrap_config_files()


    if getattr(args, "list_tools", False):
        _print_available_tools()
        sys.exit(0)

    try:
        initial_agent_name = get_initial_agent_name(args)
        model_alias = getattr(args, "model", None)
        nanny_alias = getattr(args, "nanny", None)
        config = load_config_with_cli_overrides(
            model_alias,
            nanny_alias,
            getattr(args, "nanny_brief", None),
            programmatic=args.prompt is not None,
        )

        if args.enabled_tools:
            config.enabled_tools = args.enabled_tools

        loaded_session = load_session(args, config)

        stdin_prompt = get_prompt_from_stdin()
        if args.prompt is not None:
            _run_programmatic_mode(
                args,
                config,
                loaded_session,
                stdin_prompt,
                agent_name=initial_agent_name,
                model_alias=model_alias,
                nanny_alias=nanny_alias,
            )
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

            # What the command line chose explicitly keeps winning over what a
            # resumed conversation had saved.
            agent_loop.pinned_run_state = frozenset(
                part
                for part, given in (
                    ("model", bool(model_alias)),
                    ("nanny", bool(nanny_alias)),
                    ("agent", args.agent != BuiltinAgentName.DEFAULT),
                )
                if given
            )

            if loaded_session:
                _resume_previous_session(agent_loop, *loaded_session)

            # Route to console mode or TUI based on --console flag. Imports
            # are local so console mode never loads textual.
            if getattr(args, "console", False):
                from privibe.cli.console_ui.app import run_console_ui

                run_console_ui(
                    agent_loop=agent_loop,
                    initial_prompt=args.initial_prompt or stdin_prompt,
                    show_resume_picker=args.resume is True,
                )
            else:
                from privibe.cli.textual_ui.app import StartupOptions, run_textual_ui

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
