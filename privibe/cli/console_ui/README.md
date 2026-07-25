# Console UI Mode (Thumper)

Plain text interface for privibe, alternative to the Textual TUI.

## Status: WORK IN PROGRESS

### What Works

- Basic console mode entry point (`privibe/cli/console_ui/app.py`)
- `--console` flag added to CLI
- Event streaming from AgentLoop to stdout
- Basic slash commands: `/help`, `/exit`, `/model`, `/compact`

### What Needs to Be Built

#### 1. Input Handling (~50 lines)
Replace simple `input()` with prompt_toolkit for:
- Command history (up/down arrows)
- Better Ctrl+C handling
- Windows compatibility

#### 2. Approval Callbacks (~100 lines)
When tools need permission, implement console prompts:
```
[approval needed] bash: rm -rf /tmp/cache
Type 'y' to allow, 'n' to deny, 'a' to allow all: _
```

#### 3. Question Handler (~100 lines)
Full implementation of `ask_user_question` tool:
- Display multiple questions
- Numbered choice selection
- Multi-select support
- "Other" free text option

#### 4. Command Implementations (~200 lines)
Actually implement the slash commands:
- `/model` - Show model list, let user pick, switch agent_loop
- `/compact` - Trigger conversation compaction
- `/resume` - Show session picker, load selected session
- `/config` - Edit config (maybe delegate to external editor)
- `/clear` - Clear conversation history
- Others as needed

#### 5. Tool Output Formatting (~100 lines)
Better formatting for tool results:
- File diffs (unified diff format)
- Truncated output with "..." indicator
- Markdown rendering (optional, can use rich or plain text)

#### 6. Session/Model Pickers (~100 lines)
Console menus for selecting:
- Sessions to resume
- Models to switch to

### Files Created

```
privibe/cli/console_ui/
  __init__.py
  app.py              # Main console loop (WORK IN PROGRESS)
  README.md           # This file
```

### Files Modified

```
privibe/cli/cli.py           # Added --console flag routing
privibe/cli/entrypoint.py    # Added --console argument
```

### How to Test (Current State)

```bash
privibe --console "hello"
```

This will:
1. Send "hello" to the agent
2. Stream the response to stdout
3. Enter input loop (basic, no history yet)
4. Handle `/help` and `/exit` commands

### Design Notes

- **No TUI dependencies**: Console mode doesn't import textual or rich
- **Event-driven**: Uses same AgentLoop event stream as TUI
- **Streaming**: Shows assistant content as it arrives (not waiting for completion)
- **Simple first**: Start with minimal working version, add polish later

### Next Steps

1. Add prompt_toolkit for better input handling
2. Implement approval callback
3. Implement question callback  
4. Make `/model` command actually work
5. Test on Windows
