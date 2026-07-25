# Console UI Mode (Thumper)

Plain text interface for privibe, alternative to the Textual TUI.

## Status: IMPLEMENTED

Run with:

```bash
privibe --console
privibe --console "initial prompt"
privibe --console --resume        # console session picker at startup
```

### What Works

- Interactive REPL on the same AgentLoop event seam the TUI uses
- prompt_toolkit line input (history, line editing, first-class Windows
  support; prompt() only, never full-screen mode)
- Streaming output: assistant tokens print as they arrive; reasoning is
  prefixed with [thinking] once per block
- Tool activity: call summaries via each tool's ToolUIData display,
  result lines (done / ERROR / skipped / cancelled), tool stream lines,
  compaction progress, agent profile switches
- Approval prompts: y (once) / a (always: registers the same session
  rules or tool permission as the TUI's Allow Always) / n (deny with
  optional feedback to the agent); auto_approve config respected
- ask_user_question: numbered choices, multi-select, Other free text,
  cancel; same Answer/Result models as the TUI
- Tool result bodies, TUI-parity collapsed previews: bash stdout and
  grep matches truncated to the shared tool_result_preview_lines config
  value (with "... (N more lines)"), red/green file diffs for
  file-mutating tools from the core-computed FileDiff rows (hunk budget
  = same config value), search_replace SEARCH/REPLACE blocks rendered
  as unified diff lines, todo lists with [ ]/[>]/[x]/[-] markers,
  result warnings prefixed with "!"
- Transcript replay on resume: after picking a session (or starting
  with --resume <id> / --continue) the last 20 conversation entries
  print, using the same filters core applies to resume-picker previews
  (user/assistant only, injected messages dropped, known tags
  stripped); tool calls become one-line [tool] name markers, tool
  outputs and reasoning are not replayed; /replay reprints the whole
  conversation on demand
- Commands: /help, /model (list and switch, same reload sequence as the
  TUI), /compact, /clear, /resume (console session picker), /replay,
  /log, /preview-lines (cycles 3/5/10 and persists, same config flag
  and helper the TUI uses), /exit, /quit
- Skills: /name [args] reads the skill file and sends it as the prompt,
  mirroring the TUI
- Ctrl+C interrupts a running turn and returns to the prompt; Ctrl+D or
  /exit quits

### Known Gaps / Later Polish

- Windows behavior not yet verified on a real Windows machine
- The pre-args streamed ToolCallEvent and the resolved one both print,
  so each tool call shows two [tool] lines (the TUI merges them by
  tool_call_id)
- No expand/collapse: the console always prints the TUI's collapsed
  preview; /preview-lines controls how much that shows
- No /config command; edit config.toml directly and use /model or
  restart
- No steering queue: input is read between turns, not while the agent
  is responding (TUI queues mid-turn messages)
- No ! shell passthrough

### Files

```
privibe/cli/console_ui/
  __init__.py
  app.py              # ConsoleUI: REPL, callbacks, commands
  README.md           # This file
privibe/cli/cli.py    # --console routing (asyncio.run, lazy imports)
privibe/cli/entrypoint.py  # --console argument
tests/cli/test_console_ui.py  # unit tests (events, dispatch, approvals, questions)
```

### Design Notes

- No textual import anywhere in the console path; cli.py imports the
  TUI lazily so --console never loads it
- Append-only output: every event prints once, nothing repaints, so the
  TUI's Windows render-loop cost does not exist here
- Callbacks (approval, ask_user_question) are console prompts on a
  separate prompt_toolkit session so they stay out of the input history
