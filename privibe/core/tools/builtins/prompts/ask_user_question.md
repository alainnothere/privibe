Use `ask_user_question` to gather information from the user when you need clarification, want to validate assumptions, or need help making a decision. **Don't hesitate to use this tool** — it's better to ask than to guess wrong.

## When to use

- Ambiguous instructions or unclear scope
- Technical decisions: architecture choices, library selection, tradeoffs
- Preferences: UI style, naming conventions, approach options
- Confirming understanding before starting significant work
- Several approaches could work and you want the user to pick

## Conventions

- Put the recommended option first and append "(Recommended)" to its label.
- Multiple questions display as tabs; an "Other" free-text option is added automatically unless `hide_other` is set.
- Use `multi_select` when the choices are not mutually exclusive.
- Keep labels short; put the tradeoffs in the option descriptions.
- Ask early — clarifying before starting beats redoing work after.
