Summarize the conversation so far. Your summary will REPLACE the conversation as the only remaining context, so anything you leave out is lost. Be factual and brief: state each thing once, no commentary.

Use exactly these sections:

## Goal
What the user wants, in their stated priorities. Include hard constraints they gave (e.g. "don't touch X", "no commits").

## Done so far
What was completed, and how each item was confirmed. Tag every claim:
- `(verified: tests passed)`, `(verified: command output)`, `(verified: read in file)`
- `(assumed)` or `(user said, unverified)` for anything not confirmed by tool output

Example: `Renamed offset to start_line in read_file.py (verified: tests passed).`

## Files
Each file created, changed, or central to the work: full path plus one line on what changed or why it matters. Include a short code snippet only if the next step needs it.

## Ruled out
Approaches tried and failed, hypotheses disproven — one line each on why. This prevents retrying dead ends. If nothing was ruled out, write `None.`

## Open questions
Unknowns, unresolved errors, and decisions waiting on the user. If none, write `None.`

## Next step
The single specific next action, based on the user's most recent request and the current state. Name the exact file or command where possible.

Rules:
- Never present an assumption as a fact. If unsure whether something was verified, tag it `(assumed)`.
- Do not pad. An empty section is `None.`
- Respond with ONLY the summary in the structure above.
