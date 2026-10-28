# The nanny: a second model that answers approval prompts for you

privibe asks before running a tool call that is not pre-approved: a bash
command, a file write, and so on. The nanny is a second model, picked from
your configured `[[models]]`, that answers those prompts on your behalf.

- It **approves** a call: the call runs, you see a dim one-line entry under
  the call saying why.
- It **refuses** a call: the normal approval prompt appears, with the nanny's
  reason above the options. You can allow, refuse with the nanny's reason
  (the agent reads it and goes on), or **talk to the nanny** first: explain,
  confirm, or state a one-off exception. The nanny remembers that for the
  rest of the session.

The nanny never creates session-wide permissions and never sees calls that
are hard-denied (`protected_paths`, a tool set to `never`, a "deny for this
session" rule). If it does not answer (timeout, server down, unreadable
reply) the prompt simply falls back to you with a notice.

**Escalated permissions** (a path outside the working directory under
`protect_outside_workdir`) reach you by default, nanny or not. The prompt
then has two extra rows:

- **Ask the nanny about this call** (`g`, or `5`): the nanny judges this one
  call. If it approves, the call runs; if it refuses, the prompt comes back
  with its reason, as usual.
- **Ask the nanny about calls like this for this session** (`G`, or `6`):
  the same, and every later call needing the same permissions goes to the
  nanny without stopping at you first. Nothing is granted: the nanny still
  judges each call and each refusal still reaches you.

To have the nanny take every escalated prompt from the start, set
`nanny_reviews_escalated = true`. The nanny is told which permissions are
escalated either way.

Headless (`-p`) mode and ACP do not use the nanny.

## Turning it on

In the TUI or console:

```
/ensure-commands-safety            # pick the model from a menu (TUI) or show status (console)
/ensure-commands-safety <alias>    # use that model
/ensure-commands-safety off
/ensure-commands-safety log        # the last 20 decisions
/ensure-commands-safety tell <message>   # a standing instruction, see below
```

`/nanny` is the short alias for all of these.

Or in `~/.privibe/config.toml`:

```toml
nanny_model = "reviewer"       # an alias from [[models]]; absent = off
nanny_timeout_seconds = 60     # wait for one answer before asking you instead
nanny_write_full_lines = 80    # writes up to this many lines are shown whole
nanny_reviews_escalated = false  # true: the nanny also answers escalated prompts
nanny_brief = """
This session edits a Garmin firmware dump. Reading the mounted device is
expected; writing to it, or anything under /media, is not.
"""
```

## Briefing the nanny

The nanny knows the session id, the working directory and the agent
profile, and then only what it sees call by call. Two ways to give it more:

- `nanny_brief` in the config is sent right after the session header, before
  the first call: what the task is, what to watch for, what is fine. It is
  part of the nanny's stable prefix, so it costs nothing per call.
- `/ensure-commands-safety tell <message>` (or `/nanny tell ...`) sends a
  standing instruction mid-session, outside any approval prompt: "stop
  approving curl", "pip installs are fine from now on". It waits for a
  review in flight, then goes into the nanny's conversation like your
  messages during a prompt do, and the nanny answers with what it
  understood. The latest instruction wins where two disagree.

Both are the human speaking and carry the same weight as talking to the
nanny from a refused prompt: an instruction applies until you change it.

Any alias is valid: a model on another llama-server, the same server, or
the very same model as the worker. The same model in a separate reviewer
conversation still catches a real share of mistakes (an agent mid-task
rationalises a destructive command; a fresh context with no task pressure
does not), but it shares the worker's weaknesses. For a worker you do not
trust in itself, use a different model.

`/status` shows the nanny model or `off`.

## How it works

The nanny is its own conversation, separate from the main one and never
mixed into it. It receives, as user messages, each call to judge, what you
asked the worker since the previous call, one-line outcomes of calls that
ran, and your decisions and messages. It answers with two lines:

```
VERDICT: approve | refuse | more
REASON: one short line
```

`more` is only for a file write that was shown shortened; the full content
is sent once and the nanny decides.

The nanny may think as much as it likes, out loud or in a reasoning channel.
The parser reads the visible answer first and the reasoning as a fallback,
takes the last verdict it finds (a deliberating model lands at the end),
accepts a bare "Approve." or "Refuse: ..." without the label, and fills a
missing reason from the reasoning. An answer with no verdict anywhere earns
one retry nudge; still nothing, and it counts as a refusal, so the prompt
reaches you. An unreadable answer is never an approval.

Calls are reviewed one at a time. While you decide a refused call or talk
to the nanny, other calls wait.

Everything is stored beside the session: the nanny's own transcript under
`<session>/nanny/`, and `<session>/nanny/decisions.jsonl`, one line per
reviewed call, keyed by the main conversation's tool call id. Resuming a
session resumes the nanny with it. Rewinding the main conversation does not
rewind the nanny's; it is told what was dropped.

## The prompts are yours to edit

The nanny's behaviour lives in prompt files, overridable like every other
prompt (copy the file to `~/.privibe/prompts/` or `.privibe/prompts/` in the
project and edit it):

- `nanny.md`: the system prompt. Written for a small local model that likes
  to think out loud: five questions (does it send anything off this machine,
  destroy data, change who can do what, leak secrets, reach outside the
  project), answer format only, no deliberation in the output.
- `nanny_header.md`: the first message of the session (session id, working
  directory, profile). Together with the system prompt it is the nanny's
  KV-cache identity, so keep it stable.
- `nanny_ask.md`: how one call is presented.
- `nanny_more.md`: the follow-up for a `more` verdict.
- `nanny_retry.md`: the one nudge after an answer with no readable verdict.
- `nanny_talk.md`: how your messages reach it from a refused prompt.
- `nanny_tell.md`: how `nanny_brief` and `tell` messages reach it.

## Trying it without a model

`scripts/nanny_live_demo.py` spawns the real CLI in console mode against a
scripted mock server that plays both the worker and the nanny: a harmless
command is approved silently, an upload of `.env` is refused, you talk to the
nanny, refuse with its reason, and the worker goes on. Run it from the repo
root with `uv run python scripts/nanny_live_demo.py`.
