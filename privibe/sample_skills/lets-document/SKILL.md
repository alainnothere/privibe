---
name: lets-document
description: "Document a process, investigation, system state, or finding into a living context file. 
Use this skill when the user wants to capture what is happening, why something occurred, what the current state of something is, or to build a shared understanding of a problem or system. 
This is not a planning skill — it documents reality as it is discovered.
This will not result in actions that should be pursued, so don't push the user to -implement it-."
user-invocable: true
---

# Let's Document Skill

- This skill maintains a living context file that captures what is known, what is uncertain, and how understanding has evolved. 
- The goal is to document truth — not assumptions, not what seems likely, not what the user expects to be true. If something is not confirmed, it must be marked as such.
- The model must not be a yes-man. If evidence contradicts a stated belief, the model must say so. If the user asserts something that hasn't been verified, the model must flag it as unverified. Agreement should reflect reality, not politeness.
- Passwords should not be stored, unless requested by the user.

---

## Step 1 — Locate or create the context file

- Ask the user if a context file already exists, and if so, where.
- If no file exists, propose creating one in the current working directory with a name that reflects the subject (e.g., `context-router-investigation.md`). Confirm the name and location with the user before creating it.
- If the file exists, read it fully before doing anything else. State a brief summary of what it currently contains so the user can confirm you have understood it correctly.

---

## Step 2 — Understand what is being documented

Ask the user what they want to document. It must fall into one of these categories — identify which one (there may be more than one):

- **Request** — what the user has asked for or is trying to achieve
- **State** — the current condition of a system, configuration, or environment
- **Event** — something that happened (e.g., a bug, an outage, a behavior)
- **Finding** — something discovered during investigation
- **Contradiction** — something that conflicts with a previously documented statement

Do not proceed until you know what category you are working with.

---

## Step 3 — Gather information without assuming

- Ask questions to fill gaps. Do not fill gaps yourself with assumptions.
- If the user states something as fact, ask how they know. If it was observed directly, say so. If it was inferred, say so and mark it as unverified.
- If the user says something that contradicts what is already in the context file, stop and surface the contradiction explicitly. Do not silently reconcile it. Example: *"This conflicts with what we documented earlier: [quote]. Which is correct, or do we not know yet?"*
- If something cannot be verified right now, document it as an open question, not as a fact.

---

## Step 4 — Update the context file

The context file must have the following sections. Create any that are missing. Do not remove existing sections.

### Subject
One paragraph describing what this document is about. Update this if the scope has changed.

### Current State
What is known to be true, verified, and confirmed. Each entry should note how it was verified (e.g., observed in logs, confirmed by command output, stated by user and unverified).

### Questions
Open questions that have not been answered. These are things we do not know yet. Entries are added here when something is uncertain and removed (with a note in What Changed) when they are resolved.

### Gotchas
Things discovered that were surprising, contradicted an assumption, or could cause problems. Update as needed to reflect the current status.

### What Changed
Append-only. Each entry is a single paragraph describing what was added, updated, or resolved in this revision, and why. Never edit or remove existing entries in this section.

---

## Step 5 — Confirm before writing

- Show the user the exact changes you are about to make to the context file (what is being added, updated, or moved).
- If you are resolving a question, show which question is being closed and what the answer is.
- If you are adding a gotcha, explain why it qualifies as one.
- Wait for confirmation before writing.

---

## Step 6 — Write and confirm

- Apply the changes to the context file.
- Read the updated file back and confirm it looks correct.
- State what was updated in plain language.
